"""
Local Model Shard for Split Inference.

This module loads ONLY the local portion of the model:
  - Embedding layer (embed_tokens)
  - First K transformer layers
  - LM head (lm_head)

The remaining layers are discarded to minimize local VRAM usage.
Raw user tokens are processed here and NEVER leave this server.

Reference architecture: Llama-style models (works with Llama 3.x, Qwen2, Mistral)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Tuple
import logging

from split_inference.config import ModelConfig

logger = logging.getLogger(__name__)


class LocalModelShard(nn.Module):
    """
    The local half of a split LLM.
    
    Architecture:
        Input tokens → embed_tokens → RMSNorm → Layer_0 → ... → Layer_{K-1} → [hidden_states sent to server]
        [hidden_states received from server] → RMSNorm → lm_head → logits → sample
    
    Memory footprint (Llama-3.1-8B, K=2):
        - embed_tokens: vocab_size * hidden_dim * 2B = 128256 * 4096 * 2 ≈ 1.0 GB
        - 2 transformer layers: ~2 * 0.5 GB ≈ 1.0 GB
        - lm_head: same as embed_tokens (often tied) ≈ 1.0 GB
        - Total: ~3 GB (fits on any modern GPU, even consumer cards)
    """
    
    def __init__(self, config: ModelConfig, device: str = "cuda:0"):
        super().__init__()
        self.config = config
        self.device = device
        self.model_config = None
        
        # These will be populated by load_model()
        self.embed_tokens = None
        self.local_layers = None
        self.input_norm = None       # Pre-layer norm (if applicable)
        self.final_norm = None       # Post-layer norm (for LM head input)
        self.lm_head = None
        self.rotary_emb = None       # RoPE embeddings
        
    def load_model(self):
        """
        Load the full model, extract local components, discard the rest.
        
        Strategy:
        1. Load full model in CPU memory (or use device_map="auto" for large models)
        2. Copy local layers to GPU
        3. Delete the rest to free memory
        """
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Split point: keeping first {self.config.local_layers} layers locally")
        
        # Load config first to validate
        self.model_config = AutoConfig.from_pretrained(self.config.model_name)
        
        # Load full model to CPU
        full_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.dtype),
            device_map="cpu",  # Load to CPU first
            low_cpu_mem_usage=True,
        )
        
        # Extract components (Llama-style architecture)
        # Adjust attribute names for different model families
        base_model = full_model.model  # LlamaModel inside LlamaForCausalLM
        
        # 1. Embedding layer
        self.embed_tokens = base_model.embed_tokens.to(self.device)
        
        # 2. First K transformer layers
        self.local_layers = nn.ModuleList([
            base_model.layers[i].to(self.device)
            for i in range(self.config.local_layers)
        ])
        
        # 3. Final norm (needed before LM head)
        self.final_norm = base_model.norm.to(self.device)
        
        # 4. LM head
        self.lm_head = full_model.lm_head.to(self.device)
        
        # 5. RoPE (needed for position encoding in local layers)
        if hasattr(base_model.layers[0].self_attn, 'rotary_emb'):
            self.rotary_emb = base_model.layers[0].self_attn.rotary_emb.to(self.device)
        
        # Delete full model to free CPU memory
        del full_model
        del base_model
        torch.cuda.empty_cache()
        
        local_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Local shard loaded: {local_params / 1e6:.1f}M parameters "
            f"({local_params * 2 / 1e9:.2f} GB in fp16)"
        )
    
    def forward_embed(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embedding only. Returns token embeddings before any transformer layers.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
        
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
        """
        return self.embed_tokens(input_ids)
    
    def forward_local_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run hidden states through the first K transformer layers.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            position_ids: [batch_size, seq_len]
            attention_mask: Causal attention mask
        
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim] after K layers
        """
        for layer in self.local_layers:
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # Llama layers return (hidden_states, present_kv_cache, ...)
            hidden_states = layer_output[0]
        
        return hidden_states
    
    def forward_to_split(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full local forward pass: embed → first K layers.
        Returns the activation tensor that will be sent to the remote server.
        
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim] — the "split tensor"
        """
        # Generate position IDs if not provided
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=self.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        # Embedding
        hidden_states = self.forward_embed(input_ids)
        
        # Local transformer layers
        hidden_states = self.forward_local_layers(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        
        return hidden_states
    
    def forward_lm_head(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply final norm + LM head to produce logits.
        Called AFTER receiving processed hidden states back from the server.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] from remote server
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    
    def forward_decode_step(
        self,
        token_id: torch.Tensor,
        position_id: int,
        past_key_values: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Single decode step through local layers.
        Uses KV cache for the local layers to avoid recomputation.
        
        Args:
            token_id: [1, 1] — single token
            position_id: Position in the full sequence
            past_key_values: KV cache from previous local layer forward passes
        
        Returns:
            (hidden_states [1, 1, hidden_dim], updated_past_key_values)
        """
        position_ids = torch.tensor(
            [[position_id]], dtype=torch.long, device=self.device
        )
        
        hidden_states = self.embed_tokens(token_id)
        
        new_past_key_values = []
        for i, layer in enumerate(self.local_layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            layer_output = layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=True,
            )
            hidden_states = layer_output[0]
            new_past_key_values.append(layer_output[1])  # Updated KV cache
        
        return hidden_states, new_past_key_values
    
    @torch.no_grad()
    def sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> int:
        """
        Sample next token from logits. Runs entirely on the local server.
        
        Args:
            logits: [1, 1, vocab_size] — logits for the last position
        
        Returns:
            Sampled token ID
        """
        logits = logits[:, -1, :] / temperature  # [1, vocab_size]
        
        # Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)
        
        return token_id.item()


def load_tokenizer(model_name: str):
    """Load tokenizer (stays on local server, never exposed)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

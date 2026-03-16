"""
Headless LLM for SGLang — Custom Model that starts from Hidden States.

This is the KEY integration point with SGLang. Instead of the normal flow:
    input_ids → embed_tokens → layers[0..L] → norm → lm_head → logits

We have:
    hidden_states (from gRPC) → layers[K..L] → output_hidden_states (sent back via gRPC)

The embedding and LM head are NOT loaded on this server.
SGLang handles: RadixAttention, continuous batching, KV cache, CUDA graphs.

Registration:
    This model is registered with SGLang's ModelRegistry before server launch.
    The config.json is modified to use "HeadlessLlamaForRemoteInference" as architecture.

Reference: https://docs.sglang.io/supported_models/support_new_models.html
"""
import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple

# SGLang imports (available when running inside SGLang runtime)
try:
    from sglang.srt.layers.attention.radix_attention import RadixAttention
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
    
    # Standard Llama building blocks from SGLang
    from sglang.srt.models.llama import (
        LlamaDecoderLayer,
        LlamaForCausalLM,
    )
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    # Define stubs for development outside SGLang
    class QuantizationConfig:
        pass
    class ForwardBatch:
        pass
    class PPProxyTensors:
        pass

from transformers import LlamaConfig
import logging

logger = logging.getLogger(__name__)


class HeadlessLlamaModel(nn.Module):
    """
    A Llama model that contains ONLY layers K through L (no embedding, no LM head).
    
    This is the "body" of the split model that runs on the GPU-rich main server.
    It processes intermediate activations received from the local server.
    
    Key differences from standard LlamaModel:
    1. No embed_tokens — input is already a hidden_states tensor
    2. No final norm — that's on the local server (before lm_head)
    3. Layer indices start at K, not 0 (important for RoPE and KV cache)
    4. forward() accepts hidden_states instead of input_ids
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        local_layers: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.local_layers = local_layers
        self.hidden_size = config.hidden_size
        self.num_total_layers = config.num_hidden_layers
        self.num_remote_layers = config.num_hidden_layers - local_layers
        
        # Only load layers K through L
        # IMPORTANT: layer_id must match the original model's layer indices
        # so that RoPE frequencies and KV cache slots align correctly
        if SGLANG_AVAILABLE:
            self.layers = nn.ModuleList([
                LlamaDecoderLayer(
                    config,
                    layer_id=i,  # Original layer index, not sequential from 0
                    quant_config=quant_config,
                )
                for i in range(local_layers, config.num_hidden_layers)
            ])
        else:
            # Stub for development
            self.layers = nn.ModuleList()
        
        logger.info(
            f"HeadlessLlama initialized: layers [{local_layers}..{config.num_hidden_layers}), "
            f"{self.num_remote_layers} layers, no embed/lm_head"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,       # Ignored — we use input_embeds
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        """
        SGLang-compatible forward pass starting from hidden states.
        
        In SGLang's model execution flow, ModelRunner calls:
            hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        
        For our headless model:
        - input_ids is a DUMMY tensor (required by SGLang's interface but unused)
        - input_embeds contains the actual hidden states from the local server
        - positions are the real position IDs
        
        Args:
            input_ids: Dummy tensor (SGLang requires this parameter)
            positions: Position IDs for RoPE
            forward_batch: SGLang batch info (attention masks, KV cache, etc.)
            input_embeds: The ACTUAL input — hidden states from local server
            pp_proxy_tensors: Pipeline parallel proxy tensors (if using PP)
        
        Returns:
            hidden_states: [num_tokens, hidden_dim] processed through remote layers
        """
        if input_embeds is None:
            raise ValueError(
                "HeadlessLlama requires input_embeds (hidden states from local server). "
                "input_ids alone is not supported — embedding is on the local server."
            )
        
        hidden_states = input_embeds
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                positions,
                forward_batch,
                residual,
            )
        
        # NOTE: We do NOT apply final_norm here.
        # The final norm + lm_head is on the local server.
        # We return raw hidden states from the last transformer layer.
        
        # If using residual connection pattern (Pre-LN), merge residual
        if residual is not None:
            hidden_states = hidden_states + residual
        
        return hidden_states


class HeadlessLlamaForRemoteInference(nn.Module):
    """
    Top-level SGLang-compatible model class.
    
    This wraps HeadlessLlamaModel and provides the interface SGLang expects:
    - forward() that returns LogitsProcessorOutput (even though we don't compute logits)
    - load_weights() for selective weight loading
    - Proper integration with SGLang's ModelRegistry
    
    IMPORTANT: We return hidden_states in LogitsProcessorOutput.next_token_logits
    as a hack — because SGLang's pipeline expects logits. The local server
    treats these "logits" as hidden states and applies its own LM head.
    
    A cleaner approach would be to modify SGLang's ModelRunner to support
    a "hidden_states" return mode, but that requires deeper changes.
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        local_layers: int = 2,
    ):
        super().__init__()
        self.config = config
        self.local_layers = local_layers
        
        # The headless model (no embedding, no LM head)
        self.model = HeadlessLlamaModel(
            config,
            local_layers=local_layers,
            quant_config=quant_config,
        )
        
        # NO lm_head here — it's on the local server
        # NO embed_tokens here — it's on the local server
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> "LogitsProcessorOutput":
        """
        SGLang-compatible forward pass.
        
        Returns hidden states packaged as LogitsProcessorOutput.
        The "logits" field actually contains hidden states — the local server
        knows to apply its own LM head to these.
        """
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors,
        )
        
        if SGLANG_AVAILABLE:
            # Package hidden states as if they were logits
            # This is a compatibility shim — SGLang expects LogitsProcessorOutput
            output = LogitsProcessorOutput(
                next_token_logits=hidden_states,  # Actually hidden states!
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                input_token_logprobs=None,
                input_top_logprobs=None,
                output_top_logprobs=None,
            )
            return output
        else:
            return hidden_states
    
    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        """
        Selectively load only the weights for layers K through L.
        
        SGLang calls this during model loading. We intercept and skip
        embedding/lm_head weights since they belong to the local server.
        
        Weight name mapping:
            Original: model.layers.{i}.self_attn.q_proj.weight
            Our model: model.layers.{i-K}.self_attn.q_proj.weight
        """
        params = dict(self.named_parameters())
        
        for name, loaded_weight in weights:
            # Skip embedding and LM head weights
            if "embed_tokens" in name or "lm_head" in name:
                logger.debug(f"Skipping local-only weight: {name}")
                continue
            
            # Skip final norm (it's on the local server)
            if name == "model.norm.weight":
                logger.debug(f"Skipping local-only weight: {name}")
                continue
            
            # Skip layers that belong to the local server
            if "model.layers." in name:
                # Extract layer index
                parts = name.split(".")
                layer_idx = int(parts[2])
                
                if layer_idx < self.local_layers:
                    logger.debug(f"Skipping local layer weight: {name}")
                    continue
                
                # Remap layer index: original layer i → our layer (i - K)
                new_idx = layer_idx - self.local_layers
                parts[2] = str(new_idx)
                new_name = ".".join(parts)
            else:
                new_name = name
            
            # Load the weight
            if new_name in params:
                loaded_weight = loaded_weight.to(params[new_name].dtype)
                params[new_name].data.copy_(loaded_weight)
                logger.debug(f"Loaded: {name} → {new_name}")
            else:
                logger.warning(f"Unexpected weight: {name} (mapped to {new_name})")
        
        loaded_count = sum(1 for _ in self.parameters())
        logger.info(
            f"Loaded {loaded_count} parameter tensors for remote layers "
            f"[{self.local_layers}..{self.config.num_hidden_layers})"
        )


# ============================================================================
# EntryClass for SGLang model registry
# ============================================================================

# SGLang uses this to discover the model class
EntryClass = HeadlessLlamaForRemoteInference

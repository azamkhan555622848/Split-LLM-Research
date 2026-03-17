"""
Headless Transformer for SGLang — Multi-Architecture Remote Model.

Supports Llama, Mistral, and Qwen2 families (all share the same layer structure).

Flow:
    hidden_states (from gRPC) -> layers[K..L] -> output_hidden_states (sent back via gRPC)

No embedding, no LM head — those stay on the local server.

Registration:
    Uses auto_map in config.json for process-isolation-safe model loading.
    See launch.py:prepare_headless_config() for the setup.

Reference: https://docs.sglang.io/supported_models/support_new_models.html
"""
import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple
import logging

# SGLang imports (available when running inside SGLang runtime)
try:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    class QuantizationConfig:
        pass
    class ForwardBatch:
        pass
    class PPProxyTensors:
        pass

logger = logging.getLogger(__name__)

# Architecture -> (SGLang decoder layer class import path, HF config class)
ARCH_REGISTRY = {
    "LlamaForCausalLM": {
        "decoder_layer": ("sglang.srt.models.llama", "LlamaDecoderLayer"),
        "hf_config": "LlamaConfig",
    },
    "MistralForCausalLM": {
        # Mistral reuses Llama layers in SGLang
        "decoder_layer": ("sglang.srt.models.llama", "LlamaDecoderLayer"),
        "hf_config": "MistralConfig",
    },
    "Qwen2ForCausalLM": {
        "decoder_layer": ("sglang.srt.models.qwen2", "Qwen2DecoderLayer"),
        "hf_config": "Qwen2Config",
    },
}


def _get_decoder_layer_class(architecture: str):
    """Import and return the correct decoder layer class for an architecture."""
    if architecture not in ARCH_REGISTRY:
        logger.warning(f"Unknown architecture {architecture}, falling back to Llama")
        architecture = "LlamaForCausalLM"

    info = ARCH_REGISTRY[architecture]
    module_path, class_name = info["decoder_layer"]

    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class HeadlessTransformerModel(nn.Module):
    """
    A transformer model containing ONLY layers K through L (no embedding, no LM head).

    Works with Llama, Mistral, and Qwen2 — all share the same decoder layer
    forward signature: (positions, hidden_states, forward_batch, residual) -> (hidden_states, residual)
    """

    def __init__(
        self,
        config,
        local_layers: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        architecture: str = "LlamaForCausalLM",
    ):
        super().__init__()
        self.config = config
        self.local_layers = local_layers
        self.hidden_size = config.hidden_size
        self.num_total_layers = config.num_hidden_layers
        self.num_remote_layers = config.num_hidden_layers - local_layers
        self.architecture = architecture

        if SGLANG_AVAILABLE:
            DecoderLayer = _get_decoder_layer_class(architecture)
            self.layers = nn.ModuleList([
                DecoderLayer(
                    config,
                    layer_id=i,  # Original layer index for correct RoPE/KV cache
                    quant_config=quant_config,
                )
                for i in range(local_layers, config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList()

        logger.info(
            f"HeadlessTransformer initialized: arch={architecture}, "
            f"layers [{local_layers}..{config.num_hidden_layers}), "
            f"{self.num_remote_layers} remote layers"
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
                "HeadlessTransformer requires input_embeds (hidden states from local server). "
                "input_ids alone is not supported — embedding is on the local server."
            )

        hidden_states = input_embeds
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        # Merge residual (Pre-LN pattern)
        if residual is not None:
            hidden_states = hidden_states + residual

        # NOTE: No final_norm here — that's on the local server before lm_head
        return hidden_states


class HeadlessTransformerForRemoteInference(nn.Module):
    """
    Top-level SGLang-compatible model class for multi-architecture headless inference.

    Registered via auto_map in config.json for process-isolation-safe loading.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.local_layers = getattr(config, "local_layers", 2)

        # Detect original architecture from config
        original_arch = getattr(config, "original_architecture", "LlamaForCausalLM")

        self.model = HeadlessTransformerModel(
            config,
            local_layers=self.local_layers,
            quant_config=quant_config,
            architecture=original_arch,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> "LogitsProcessorOutput":
        """SGLang-compatible forward pass. Returns hidden states as LogitsProcessorOutput."""
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors,
        )

        if SGLANG_AVAILABLE:
            return LogitsProcessorOutput(
                next_token_logits=hidden_states,  # Actually hidden states
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                input_token_logprobs=None,
                input_top_logprobs=None,
                output_top_logprobs=None,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Selectively load only weights for remote layers K..L.

        Skips: embed_tokens, lm_head, final norm, and layers 0..K-1.
        Remaps layer indices: original layer i -> our layer (i - K).
        """
        params = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Skip local-only components
            if any(skip in name for skip in ("embed_tokens", "lm_head", "model.norm.")):
                continue

            # Handle layer index remapping
            if "model.layers." in name:
                parts = name.split(".")
                layer_idx = int(parts[2])

                if layer_idx < self.local_layers:
                    continue

                # Remap: original layer i -> our layer (i - K)
                parts[2] = str(layer_idx - self.local_layers)
                new_name = ".".join(parts)
            else:
                new_name = name

            if new_name in params:
                loaded_weight = loaded_weight.to(params[new_name].dtype)
                params[new_name].data.copy_(loaded_weight)
            else:
                logger.warning(f"Unexpected weight: {name} (mapped to {new_name})")

        loaded_count = sum(1 for _ in self.parameters())
        logger.info(
            f"Loaded {loaded_count} parameter tensors for remote layers "
            f"[{self.local_layers}..{self.config.num_hidden_layers})"
        )


# Backwards compatibility alias
HeadlessLlamaForRemoteInference = HeadlessTransformerForRemoteInference

# SGLang EntryClass
EntryClass = HeadlessTransformerForRemoteInference

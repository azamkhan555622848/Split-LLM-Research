"""
Shared configuration for Privacy-Preserving Split LLM Inference.
"""
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Architectures that share the Llama-style layer structure
# (model.layers, model.embed_tokens, model.norm, lm_head)
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
}


@dataclass
class ModelConfig:
    """Model split configuration."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    architecture: str = "LlamaForCausalLM"  # Detected from HF config
    total_layers: int = 32          # Total transformer layers in the model
    local_layers: int = 2           # Number of layers kept on local server (split point K)
    hidden_dim: int = 4096          # Hidden dimension of the model
    num_heads: int = 32             # Number of attention heads
    num_kv_heads: int = 8           # Number of key-value heads (GQA)
    head_dim: int = 128             # Dimension per head
    vocab_size: int = 128256        # Vocabulary size
    max_seq_len: int = 4096         # Maximum sequence length
    dtype: str = "float16"          # Model dtype

    @classmethod
    def from_pretrained(cls, model_name: str, local_layers: int = 2, dtype: str = "float16") -> "ModelConfig":
        """Auto-detect model parameters from HuggingFace config."""
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)

        arch = "LlamaForCausalLM"
        if hasattr(hf_config, "architectures") and hf_config.architectures:
            arch = hf_config.architectures[0]
            if arch not in SUPPORTED_ARCHITECTURES:
                logger.warning(f"Architecture {arch} not in supported set, proceeding anyway")

        hidden_dim = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", None) or (hidden_dim // num_heads)
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)

        config = cls(
            model_name=model_name,
            architecture=arch,
            total_layers=hf_config.num_hidden_layers,
            local_layers=local_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=hf_config.vocab_size,
            max_seq_len=getattr(hf_config, "max_position_embeddings", 4096),
            dtype=dtype,
        )
        logger.info(
            f"ModelConfig.from_pretrained: {arch}, {config.total_layers} layers, "
            f"hidden={hidden_dim}, heads={num_heads}/{num_kv_heads}, vocab={config.vocab_size}"
        )
        return config


@dataclass
class PrivacyConfig:
    """Differential privacy and encryption configuration."""
    # Differential Privacy
    dp_enabled: bool = True
    dp_epsilon: float = 8.0         # Privacy budget (lower = more private, noisier)
    dp_delta: float = 1e-5          # Failure probability
    dp_sensitivity: float = 1.0     # L2 sensitivity of activations (calibrate empirically)
    dp_mechanism: str = "gaussian"  # "gaussian" or "laplace"

    # Activation perturbation (structured noise)
    perturbation_enabled: bool = False
    perturbation_seed: int = 42     # Shared seed for reversible perturbation
    perturbation_scale: float = 0.1

    # Clipping (bound activation norms before adding noise)
    clip_norm: float = 10.0         # Max L2 norm per activation vector
    clip_enabled: bool = True


@dataclass
class NetworkConfig:
    """gRPC and TLS configuration."""
    main_server_host: str = "0.0.0.0"
    main_server_port: int = 50051
    main_server_address: str = "localhost:50051"  # Address local server connects to

    # mTLS certificates
    tls_enabled: bool = True
    ca_cert: str = "certs/ca.pem"
    server_cert: str = "certs/server.pem"
    server_key: str = "certs/server.key"
    client_cert: str = "certs/client.pem"
    client_key: str = "certs/client.key"

    # gRPC settings
    max_message_size: int = 64 * 1024 * 1024  # 64MB (large activation tensors)
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000
    compression: str = "gzip"       # gRPC compression for activation tensors


@dataclass
class SGLangConfig:
    """SGLang-specific configuration for the main server."""
    sglang_port: int = 30000
    tp_size: int = 1                # Tensor parallelism degree
    dp_size: int = 1                # Data parallelism degree (SGLang DP, not differential privacy)
    mem_fraction: float = 0.85      # GPU memory fraction for KV cache
    max_running_requests: int = 64  # Max concurrent requests
    chunked_prefill_size: int = 8192
    enable_radix_cache: bool = True
    quantization: Optional[str] = None  # "fp8", "awq", "gptq", etc.


@dataclass
class SplitInferenceConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)

    # Speculative decoding for latency amortization
    speculative_enabled: bool = False
    speculative_lookahead_k: int = 3  # Jacobi iteration depth

    # Batching
    prefill_batch_size: int = 1     # How many prompts to prefill at once

    @property
    def remote_layers(self) -> int:
        return self.model.total_layers - self.model.local_layers

    @property
    def remote_layer_range(self) -> tuple:
        return (self.model.local_layers, self.model.total_layers)

    def validate(self):
        """Validate configuration ranges. Raises ValueError on invalid config."""
        m = self.model
        if m.local_layers < 1:
            raise ValueError(f"local_layers must be >= 1, got {m.local_layers}")
        if m.local_layers >= m.total_layers:
            raise ValueError(
                f"local_layers ({m.local_layers}) must be < total_layers ({m.total_layers})"
            )
        if m.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {m.hidden_dim}")
        if m.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {m.vocab_size}")

        p = self.privacy
        if p.dp_epsilon <= 0:
            raise ValueError(f"dp_epsilon must be > 0, got {p.dp_epsilon}")
        if p.dp_delta <= 0 or p.dp_delta >= 1:
            raise ValueError(f"dp_delta must be in (0, 1), got {p.dp_delta}")
        if p.clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0, got {p.clip_norm}")
        if p.dp_mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"dp_mechanism must be 'gaussian' or 'laplace', got {p.dp_mechanism}")

        n = self.network
        if n.main_server_port < 1 or n.main_server_port > 65535:
            raise ValueError(f"main_server_port must be 1-65535, got {n.main_server_port}")

        s = self.sglang
        if s.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {s.tp_size}")
        if s.mem_fraction <= 0 or s.mem_fraction > 1:
            raise ValueError(f"mem_fraction must be in (0, 1], got {s.mem_fraction}")

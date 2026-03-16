"""
Shared configuration for Privacy-Preserving Split LLM Inference.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model split configuration."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    total_layers: int = 32          # Total transformer layers in the model
    local_layers: int = 2           # Number of layers kept on local server (split point K)
    hidden_dim: int = 4096          # Hidden dimension of the model
    num_heads: int = 32             # Number of attention heads
    head_dim: int = 128             # Dimension per head
    vocab_size: int = 128256        # Vocabulary size
    max_seq_len: int = 4096         # Maximum sequence length
    dtype: str = "float16"          # Model dtype


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
    max_num_reqs: int = 64          # Max concurrent requests
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

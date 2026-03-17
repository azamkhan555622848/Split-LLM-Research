"""
Local Server gRPC Client.

Orchestrates the full inference loop:
1. Tokenize user input (local)
2. Forward through embedding + first K layers (local)
3. Apply privacy protections (clip, DP noise, perturbation)
4. Send encrypted activations to main server via gRPC
5. Receive processed activations back
6. Run LM head to get logits (local)
7. Sample next token (local)
8. Repeat for autoregressive decoding

The user's raw text, tokens, and final output NEVER leave this process.
"""
import os
import time
import struct
import logging
import argparse
import numpy as np
import torch
import grpc
from typing import Optional, Generator

from split_inference.config import SplitInferenceConfig, PrivacyConfig, ModelConfig
from split_inference.local_server.local_model import LocalModelShard, load_tokenizer
from split_inference.local_server.privacy_engine import PrivacyEngine

from split_inference.proto import split_inference_pb2 as pb2
from split_inference.proto import split_inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class ActivationSerializer:
    """
    Serialize/deserialize torch tensors for gRPC transport.
    Uses raw bytes with metadata header for zero-copy efficiency.
    """
    
    # Map dtype string -> (torch dtype for computation, torch dtype for wire, numpy dtype for wire)
    DTYPE_MAP = {
        "float16": (torch.float16, np.float16),
        "bfloat16": (torch.float16, np.float16),  # bfloat16 -> fp16 on wire (no numpy bf16)
        "float32": (torch.float32, np.float32),
    }

    @staticmethod
    def serialize(tensor: torch.Tensor, dtype: str = "float16") -> bytes:
        """
        Serialize a tensor to bytes.
        Format: [ndim (1 byte)] + [shape_dims (4 bytes each)] + [raw data]
        """
        wire_torch_dtype, _ = ActivationSerializer.DTYPE_MAP.get(
            dtype, (torch.float16, np.float16)
        )
        t = tensor.detach().cpu().contiguous()
        if t.dtype != wire_torch_dtype:
            t = t.to(wire_torch_dtype)

        ndim_header = struct.pack('B', len(t.shape))
        shape_header = struct.pack(f'{len(t.shape)}I', *t.shape)

        return ndim_header + shape_header + t.numpy().tobytes()

    @staticmethod
    def deserialize(data: bytes, dtype: str = "float16", device: str = "cuda:0") -> torch.Tensor:
        """Deserialize bytes back to a tensor."""
        _, wire_np_dtype = ActivationSerializer.DTYPE_MAP.get(
            dtype, (torch.float16, np.float16)
        )
        ndim = struct.unpack('B', data[:1])[0]
        shape_size = ndim * 4
        shape = struct.unpack(f'{ndim}I', data[1:1 + shape_size])
        tensor_data = data[1 + shape_size:]
        arr = np.frombuffer(tensor_data, dtype=wire_np_dtype).reshape(shape)

        return torch.from_numpy(arr.copy()).to(device)


_RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff: 1s, 2s, 4s


def _retry_rpc(func, *args, max_retries=3, **kwargs):
    """Retry an RPC call with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            last_error = e
            if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                logger.warning(
                    f"RPC failed ({e.code().name}), retry {attempt+1}/{max_retries} in {delay}s"
                )
                time.sleep(delay)
            else:
                raise  # Non-retryable error
    raise last_error


class SplitInferenceClient:
    """
    The main client class that runs on the local server.
    Manages the full split inference lifecycle.

    Supports context manager protocol for automatic cleanup:
        with SplitInferenceClient(config) as client:
            client.connect()
            ...
    """

    def __init__(self, config: SplitInferenceConfig):
        self.config = config
        self.serializer = ActivationSerializer()

        # Load local model shard
        self.model = LocalModelShard(config.model, device="cuda:0")
        self.model.load_model()
        self.model.eval()

        # Load tokenizer
        self.tokenizer = load_tokenizer(config.model.model_name)

        # Initialize privacy engine
        self.privacy = PrivacyEngine(config.privacy, config.model.hidden_dim)

        # gRPC channel (will be initialized in connect())
        self.channel = None
        self.stub = None
        self.session_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Close the gRPC channel and clean up resources."""
        if self.channel is not None:
            try:
                self.channel.close()
            except Exception:
                pass
            self.channel = None
            self.stub = None
            logger.info("gRPC channel closed")
    
    def connect(self):
        """Establish mTLS gRPC connection to the main server."""
        net = self.config.network
        
        if net.tls_enabled:
            # Load TLS certificates for mutual authentication
            with open(net.ca_cert, 'rb') as f:
                ca_cert = f.read()
            with open(net.client_cert, 'rb') as f:
                client_cert = f.read()
            with open(net.client_key, 'rb') as f:
                client_key = f.read()
            
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert,
            )
            
            options = [
                ('grpc.max_send_message_length', net.max_message_size),
                ('grpc.max_receive_message_length', net.max_message_size),
                ('grpc.keepalive_time_ms', net.keepalive_time_ms),
                ('grpc.keepalive_timeout_ms', net.keepalive_timeout_ms),
                ('grpc.default_compression_algorithm',
                 grpc.Compression.Gzip if net.compression == "gzip" else grpc.Compression.NoCompression),
            ]
            
            self.channel = grpc.secure_channel(
                net.main_server_address, credentials, options=options,
            )
        else:
            # Insecure channel (development only!)
            logger.warning("⚠️  Using insecure gRPC channel — for development only!")
            self.channel = grpc.insecure_channel(
                net.main_server_address,
                options=[
                    ('grpc.max_send_message_length', net.max_message_size),
                    ('grpc.max_receive_message_length', net.max_message_size),
                ],
            )
        
        # Create stub
        self.stub = pb2_grpc.SplitInferenceServiceStub(self.channel)

        # Wait for channel to be ready (with timeout)
        try:
            grpc.channel_ready_future(self.channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            logger.warning("Channel not ready after 10s, proceeding anyway")

        logger.info(f"Connected to main server at {net.main_server_address}")
    
    def create_session(self) -> str:
        """Create a new inference session on the main server."""
        request = pb2.CreateSessionRequest(
            model_name=self.config.model.model_name,
            local_layers=self.config.model.local_layers,
            max_seq_len=self.config.model.max_seq_len,
            dp_metadata=pb2.DPMetadata(
                dp_enabled=self.config.privacy.dp_enabled,
                epsilon=self.config.privacy.dp_epsilon,
                delta=self.config.privacy.dp_delta,
                mechanism=self.config.privacy.dp_mechanism,
                clip_norm=self.config.privacy.clip_norm,
                perturbation_enabled=self.config.privacy.perturbation_enabled,
                perturbation_seed=self.config.privacy.perturbation_seed,
                perturbation_scale=self.config.privacy.perturbation_scale,
            ),
        )
        response = _retry_rpc(self.stub.CreateSession, request, timeout=30)
        if not response.success:
            raise RuntimeError(f"CreateSession rejected: {response.error_message}")
        self.session_id = response.session_id
        logger.info(f"Session created: {self.session_id}")
        return self.session_id
    
    @torch.no_grad()
    def _prefill(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefill phase: process full prompt through local → remote → local.
        
        Args:
            input_ids: [1, seq_len] tokenized prompt
        
        Returns:
            logits: [1, seq_len, vocab_size]
        """
        # 1. Local forward: embed + first K layers
        t0 = time.perf_counter()
        hidden_states = self.model.forward_to_split(input_ids)
        local_time = (time.perf_counter() - t0) * 1000
        
        # 2. Privacy protection
        t0 = time.perf_counter()
        protected_hidden, sigma = self.privacy.protect(hidden_states, step=0)
        privacy_time = (time.perf_counter() - t0) * 1000
        
        # 3. Serialize and send to remote server
        t0 = time.perf_counter()
        hidden_bytes = self.serializer.serialize(protected_hidden, dtype=self.config.model.dtype)

        seq_len = input_ids.shape[1]
        position_ids = list(range(seq_len))

        request = pb2.PrefillRequest(
            session_id=self.session_id,
            hidden_states=hidden_bytes,
            position_ids=position_ids,
            seq_len=seq_len,
            hidden_dim=self.config.model.hidden_dim,
            dtype=self.config.model.dtype,
            noise_sigma=sigma,
        )
        response = _retry_rpc(self.stub.Prefill, request, timeout=30)
        if not response.success:
            raise RuntimeError(f"Prefill failed: {response.error_message}")
        remote_hidden = self.serializer.deserialize(
            response.hidden_states, dtype=self.config.model.dtype,
        )

        network_time = (time.perf_counter() - t0) * 1000
        
        # 4. LM head (local)
        logits = self.model.forward_lm_head(remote_hidden)
        
        logger.debug(
            f"Prefill: local={local_time:.1f}ms, "
            f"privacy={privacy_time:.1f}ms, "
            f"network+remote={network_time:.1f}ms, "
            f"sigma={sigma:.4f}"
        )
        
        return logits
    
    @torch.no_grad()
    def _decode_step(
        self,
        token_id: torch.Tensor,
        position_id: int,
        past_key_values: Optional[list],
        step: int,
    ) -> tuple:
        """
        Single autoregressive decode step.
        
        Returns:
            (logits, updated_past_key_values)
        """
        # 1. Local forward (single token through local layers with KV cache)
        hidden_states, past_key_values = self.model.forward_decode_step(
            token_id, position_id, past_key_values,
        )
        
        # 2. Privacy protection
        protected_hidden, sigma = self.privacy.protect(hidden_states, step=step)
        
        # 3. Send to remote, receive back
        hidden_bytes = self.serializer.serialize(protected_hidden, dtype=self.config.model.dtype)

        request = pb2.DecodeRequest(
            session_id=self.session_id,
            hidden_states=hidden_bytes,
            position_id=position_id,
            hidden_dim=self.config.model.hidden_dim,
            dtype=self.config.model.dtype,
            decode_step=step,
            noise_sigma=sigma,
        )
        response = _retry_rpc(self.stub.Decode, request, timeout=10)
        if not response.success:
            raise RuntimeError(f"Decode failed: {response.error_message}")
        remote_hidden = self.serializer.deserialize(
            response.hidden_states, dtype=self.config.model.dtype,
        )
        
        # 4. LM head (local)
        logits = self.model.forward_lm_head(remote_hidden)
        
        return logits, past_key_values
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
    ) -> Generator[str, None, None] if True else str:
        """
        Full generation loop with split inference.
        
        Privacy guarantee: The main server NEVER sees:
        - Raw text
        - Token IDs  
        - Final output text
        - Logits / probabilities
        
        It only sees noisy intermediate activations.
        
        Args:
            prompt: User's input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling
            stream: If True, yield tokens as they're generated
        
        Yields/Returns:
            Generated text
        """
        # Tokenize (local only)
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to("cuda:0")
        
        seq_len = input_ids.shape[1]
        
        # ---- Prefill phase ----
        logits = self._prefill(input_ids)
        
        # Sample first token from the last position's logits
        next_token = self.model.sample_token(
            logits, temperature, top_p, top_k,
        )
        
        generated_tokens = [next_token]
        past_key_values = None  # Local KV cache
        
        if stream:
            yield self.tokenizer.decode([next_token])
        
        # ---- Decode loop ----
        for step in range(1, max_new_tokens):
            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            
            token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device="cuda:0"
            )
            position_id = seq_len + step - 1
            
            logits, past_key_values = self._decode_step(
                token_tensor, position_id, past_key_values, step,
            )
            
            next_token = self.model.sample_token(
                logits, temperature, top_p, top_k,
            )
            generated_tokens.append(next_token)
            
            if stream:
                yield self.tokenizer.decode([next_token])
        
        # Final privacy report
        report = self.privacy.get_privacy_report()
        logger.info(
            f"Generation complete: {len(generated_tokens)} tokens, "
            f"total ε={report['total_epsilon']:.2f}"
        )
        
        if not stream:
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run the local inference client."""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Split Inference — Local Client"
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("SPLIT_INFERENCE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        help="HuggingFace model path or ID",
    )
    parser.add_argument(
        "--local-layers", type=int,
        default=int(os.environ.get("SPLIT_INFERENCE_LOCAL_LAYERS", "2")),
        help="Number of layers kept locally (split point K)",
    )
    parser.add_argument(
        "--server-address", type=str,
        default=os.environ.get("SPLIT_INFERENCE_SERVER", "localhost:50051"),
        help="Main server gRPC address",
    )
    parser.add_argument(
        "--dp-epsilon", type=float,
        default=float(os.environ.get("SPLIT_INFERENCE_EPSILON", "8.0")),
        help="Differential privacy epsilon budget",
    )
    parser.add_argument(
        "--tls-enabled", action="store_true",
        default=os.environ.get("SPLIT_INFERENCE_TLS", "").lower() in ("1", "true", "yes"),
        help="Enable mTLS for gRPC",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens to generate per response",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = SplitInferenceConfig()
    config.model.model_name = args.model
    config.model.local_layers = args.local_layers
    config.privacy.dp_epsilon = args.dp_epsilon
    config.privacy.dp_enabled = True
    config.network.main_server_address = args.server_address
    config.network.tls_enabled = args.tls_enabled

    with SplitInferenceClient(config) as client:
        client.connect()
        client.create_session()

        print(f"\nPrivacy-Preserving Split Inference")
        print(f"  Model: {config.model.model_name}")
        print(f"  Split: {config.model.local_layers} local layers")
        print(f"  DP: epsilon={config.privacy.dp_epsilon}, delta={config.privacy.dp_delta}")
        print(f"  Server: {args.server_address} ({'mTLS' if args.tls_enabled else 'insecure'})")
        print(f"  Raw data stays local. Only noisy activations leave.\n")

        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break

            print("Assistant: ", end="", flush=True)
            for token in client.generate(prompt, max_new_tokens=args.max_tokens, stream=True):
                print(token, end="", flush=True)
            print()

            report = client.privacy.get_privacy_report()
            print(f"  [epsilon={report['total_epsilon']:.2f}, steps={report['total_steps']}]")


if __name__ == "__main__":
    main()

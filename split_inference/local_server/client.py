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
import time
import struct
import logging
import numpy as np
import torch
import grpc
from typing import Optional, Generator

from split_inference.config import SplitInferenceConfig, PrivacyConfig, ModelConfig
from split_inference.local_server.local_model import LocalModelShard, load_tokenizer
from split_inference.local_server.privacy_engine import PrivacyEngine

# Import generated protobuf classes (after running protoc)
# from proto import split_inference_pb2 as pb2
# from proto import split_inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class ActivationSerializer:
    """
    Serialize/deserialize torch tensors for gRPC transport.
    Uses raw bytes with metadata header for zero-copy efficiency.
    """
    
    @staticmethod
    def serialize(tensor: torch.Tensor) -> bytes:
        """
        Serialize a tensor to bytes.
        Format: [shape_dims (4 bytes each)] + [raw fp16 data]
        """
        # Ensure contiguous fp16
        t = tensor.detach().cpu().contiguous()
        if t.dtype != torch.float16:
            t = t.to(torch.float16)
        
        # Pack shape as header
        shape_header = struct.pack(f'{len(t.shape)}I', *t.shape)
        ndim_header = struct.pack('B', len(t.shape))
        
        return ndim_header + shape_header + t.numpy().tobytes()
    
    @staticmethod
    def deserialize(data: bytes, device: str = "cuda:0") -> torch.Tensor:
        """Deserialize bytes back to a tensor."""
        # Read ndim
        ndim = struct.unpack('B', data[:1])[0]
        
        # Read shape
        shape_size = ndim * 4
        shape = struct.unpack(f'{ndim}I', data[1:1 + shape_size])
        
        # Read tensor data
        tensor_data = data[1 + shape_size:]
        arr = np.frombuffer(tensor_data, dtype=np.float16).reshape(shape)
        
        return torch.from_numpy(arr.copy()).to(device)


class SplitInferenceClient:
    """
    The main client class that runs on the local server.
    Manages the full split inference lifecycle.
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
        # self.stub = pb2_grpc.SplitInferenceServiceStub(self.channel)
        logger.info(f"Connected to main server at {net.main_server_address}")
    
    def create_session(self) -> str:
        """Create a new inference session on the main server."""
        # request = pb2.CreateSessionRequest(
        #     model_name=self.config.model.model_name,
        #     local_layers=self.config.model.local_layers,
        #     max_seq_len=self.config.model.max_seq_len,
        #     dp_metadata=pb2.DPMetadata(
        #         dp_enabled=self.config.privacy.dp_enabled,
        #         epsilon=self.config.privacy.dp_epsilon,
        #         delta=self.config.privacy.dp_delta,
        #         mechanism=self.config.privacy.dp_mechanism,
        #         clip_norm=self.config.privacy.clip_norm,
        #         perturbation_enabled=self.config.privacy.perturbation_enabled,
        #         perturbation_seed=self.config.privacy.perturbation_seed,
        #         perturbation_scale=self.config.privacy.perturbation_scale,
        #     ),
        # )
        # response = self.stub.CreateSession(request)
        # self.session_id = response.session_id
        # return self.session_id
        pass  # Placeholder until protobuf is compiled
    
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
        hidden_bytes = self.serializer.serialize(protected_hidden)
        
        seq_len = input_ids.shape[1]
        position_ids = list(range(seq_len))
        
        # gRPC call (placeholder)
        # request = pb2.PrefillRequest(
        #     session_id=self.session_id,
        #     hidden_states=hidden_bytes,
        #     position_ids=position_ids,
        #     seq_len=seq_len,
        #     hidden_dim=self.config.model.hidden_dim,
        #     dtype=self.config.model.dtype,
        #     noise_sigma=sigma,
        # )
        # response = self.stub.Prefill(request)
        # remote_hidden = self.serializer.deserialize(response.hidden_states)
        
        # MOCK for development: pass through without remote
        remote_hidden = protected_hidden  # Replace with actual gRPC response
        
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
        hidden_bytes = self.serializer.serialize(protected_hidden)
        
        # gRPC call (placeholder)
        # request = pb2.DecodeRequest(
        #     session_id=self.session_id,
        #     hidden_states=hidden_bytes,
        #     position_id=position_id,
        #     hidden_dim=self.config.model.hidden_dim,
        #     dtype=self.config.model.dtype,
        #     decode_step=step,
        #     noise_sigma=sigma,
        # )
        # response = self.stub.Decode(request)
        # remote_hidden = self.serializer.deserialize(response.hidden_states)
        
        remote_hidden = protected_hidden  # MOCK
        
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
    logging.basicConfig(level=logging.INFO)
    
    config = SplitInferenceConfig()
    
    # Adjust for your setup
    config.model.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    config.model.local_layers = 2
    config.privacy.dp_epsilon = 8.0
    config.privacy.dp_enabled = True
    config.network.tls_enabled = False  # Set True in production!
    
    client = SplitInferenceClient(config)
    client.connect()
    client.create_session()
    
    # Interactive loop
    print("\n🔒 Privacy-Preserving Split Inference")
    print(f"   Model: {config.model.model_name}")
    print(f"   Split: {config.model.local_layers} local layers")
    print(f"   DP: ε={config.privacy.dp_epsilon}, δ={config.privacy.dp_delta}")
    print(f"   Raw data stays local. Only noisy activations leave.\n")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() in ("quit", "exit", "q"):
            break
        
        print("Assistant: ", end="", flush=True)
        for token in client.generate(prompt, stream=True):
            print(token, end="", flush=True)
        print()
        
        # Show privacy budget
        report = client.privacy.get_privacy_report()
        print(f"  [ε={report['total_epsilon']:.2f}, "
              f"steps={report['total_steps']}]")


if __name__ == "__main__":
    main()

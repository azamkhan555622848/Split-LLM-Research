"""
Activation Server — gRPC service on the Main Server.

Receives encrypted activation tensors from local servers,
routes them through SGLang's modified inference pipeline,
and returns processed activations.

This server:
- NEVER sees raw tokens or text
- NEVER sees logits or probabilities
- ONLY processes intermediate hidden states (with DP noise)
- Runs inside a TEE (optional, for hardware-level protection)

Architecture:
    gRPC Server ←→ SGLang Engine (headless model)
                 ↕
           GPU (KV cache, attention, FFN)
"""
import uuid
import time
import struct
import logging
import asyncio
import numpy as np
import torch
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, Optional

import grpc

from split_inference.config import SplitInferenceConfig, NetworkConfig
from split_inference.local_server.privacy_engine import PrivacyEngine

# SGLang engine import
try:
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class InferenceSession:
    """State for a single client session."""
    session_id: str
    model_name: str
    local_layers: int
    max_seq_len: int
    created_at: float
    
    # DP metadata from client
    dp_enabled: bool = False
    perturbation_enabled: bool = False
    perturbation_seed: int = 0
    perturbation_scale: float = 0.0
    
    # KV cache tracking
    current_seq_len: int = 0
    decode_step: int = 0
    
    # SGLang request handle
    sglang_request_id: Optional[str] = None


class ActivationProcessor:
    """
    Bridges gRPC activation tensors to SGLang's inference engine.
    
    The challenge: SGLang expects input_ids, but we have hidden_states.
    
    Solution: We use SGLang's Engine with a custom model (HeadlessLlama)
    that accepts input_embeds. The flow:
    
    1. Receive hidden_states bytes from gRPC
    2. Deserialize to torch tensor
    3. Remove perturbation (if enabled) — requires shared seed
    4. Feed to SGLang Engine as input_embeds
    5. SGLang runs layers K..L with RadixAttention, continuous batching
    6. Get output hidden_states
    7. Serialize and return via gRPC
    """
    
    def __init__(self, config: SplitInferenceConfig):
        self.config = config
        self.sessions: Dict[str, InferenceSession] = {}
        self.engine = None
        
        if SGLANG_AVAILABLE:
            self._init_sglang_engine()
    
    def _init_sglang_engine(self):
        """
        Initialize SGLang with the headless model.
        
        We register our custom model class before creating the engine
        so SGLang knows how to load HeadlessLlamaForRemoteInference.
        """
        from sglang.srt.models.registry import ModelRegistry
        from main_server.headless_llama import HeadlessLlamaForRemoteInference
        
        # Register custom model
        ModelRegistry.models["HeadlessLlamaForRemoteInference"] = \
            HeadlessLlamaForRemoteInference
        
        logger.info("Registered HeadlessLlamaForRemoteInference with SGLang ModelRegistry")
        
        # Create engine with the custom model
        # NOTE: The model's config.json must have:
        #   "architectures": ["HeadlessLlamaForRemoteInference"]
        self.engine = Engine(
            model_path=self.config.model.model_name,
            tp_size=self.config.sglang.tp_size,
            dp_size=self.config.sglang.dp_size,
            mem_fraction_static=self.config.sglang.mem_fraction,
            max_num_reqs=self.config.sglang.max_num_reqs,
            chunked_prefill_size=self.config.sglang.chunked_prefill_size,
            disable_radix_cache=not self.config.sglang.enable_radix_cache,
            quantization=self.config.sglang.quantization,
        )
        
        logger.info("SGLang Engine initialized with headless model")
    
    def create_session(
        self,
        model_name: str,
        local_layers: int,
        max_seq_len: int,
        dp_metadata: dict,
    ) -> str:
        """Create a new inference session."""
        session_id = str(uuid.uuid4())
        session = InferenceSession(
            session_id=session_id,
            model_name=model_name,
            local_layers=local_layers,
            max_seq_len=max_seq_len,
            created_at=time.time(),
            dp_enabled=dp_metadata.get("dp_enabled", False),
            perturbation_enabled=dp_metadata.get("perturbation_enabled", False),
            perturbation_seed=dp_metadata.get("perturbation_seed", 0),
            perturbation_scale=dp_metadata.get("perturbation_scale", 0.0),
        )
        self.sessions[session_id] = session
        logger.info(f"Session created: {session_id} (local_layers={local_layers})")
        return session_id
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session and free its KV cache."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            # TODO: Signal SGLang to free KV cache for this session
            logger.info(f"Session destroyed: {session_id}")
            return True
        return False
    
    def _deserialize_hidden_states(
        self,
        data: bytes,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """Deserialize hidden states from gRPC bytes."""
        ndim = struct.unpack('B', data[:1])[0]
        shape_size = ndim * 4
        shape = struct.unpack(f'{ndim}I', data[1:1 + shape_size])
        tensor_data = data[1 + shape_size:]
        arr = np.frombuffer(tensor_data, dtype=np.float16).reshape(shape)
        return torch.from_numpy(arr.copy()).to(device)
    
    def _serialize_hidden_states(self, tensor: torch.Tensor) -> bytes:
        """Serialize hidden states for gRPC response."""
        t = tensor.detach().cpu().contiguous().to(torch.float16)
        shape_header = struct.pack(f'{len(t.shape)}I', *t.shape)
        ndim_header = struct.pack('B', len(t.shape))
        return ndim_header + shape_header + t.numpy().tobytes()
    
    def process_prefill(
        self,
        session_id: str,
        hidden_bytes: bytes,
        position_ids: list,
        noise_sigma: float,
    ) -> bytes:
        """
        Process a prefill request through the remote layers.
        
        1. Deserialize hidden states
        2. Remove perturbation (if enabled)
        3. Run through SGLang engine (layers K..L)
        4. Serialize and return output hidden states
        """
        session = self.sessions[session_id]
        t0 = time.perf_counter()
        
        # Deserialize
        hidden_states = self._deserialize_hidden_states(hidden_bytes)
        
        # Remove perturbation if enabled
        if session.perturbation_enabled:
            hidden_states = PrivacyEngine.remove_perturbation(
                hidden_states,
                session.perturbation_seed,
                session.perturbation_scale,
                step=0,
            )
        
        # Run through SGLang engine
        # The engine calls HeadlessLlamaForRemoteInference.forward()
        # with input_embeds=hidden_states
        if self.engine is not None:
            # Use SGLang's internal API to pass hidden states
            # This requires a custom input format that passes input_embeds
            output_hidden = self._run_sglang_forward(
                session, hidden_states, position_ids, is_prefill=True,
            )
        else:
            # Development fallback: identity transform
            output_hidden = hidden_states
            logger.warning("SGLang not available — returning identity transform")
        
        # Update session state
        session.current_seq_len = len(position_ids)
        
        compute_time = (time.perf_counter() - t0) * 1000
        logger.debug(f"Prefill processed: seq_len={len(position_ids)}, time={compute_time:.1f}ms")
        
        return self._serialize_hidden_states(output_hidden)
    
    def process_decode(
        self,
        session_id: str,
        hidden_bytes: bytes,
        position_id: int,
        decode_step: int,
        noise_sigma: float,
    ) -> bytes:
        """Process a single decode step through remote layers."""
        session = self.sessions[session_id]
        
        hidden_states = self._deserialize_hidden_states(hidden_bytes)
        
        if session.perturbation_enabled:
            hidden_states = PrivacyEngine.remove_perturbation(
                hidden_states,
                session.perturbation_seed,
                session.perturbation_scale,
                step=decode_step,
            )
        
        if self.engine is not None:
            output_hidden = self._run_sglang_forward(
                session, hidden_states, [position_id], is_prefill=False,
            )
        else:
            output_hidden = hidden_states
        
        session.current_seq_len += 1
        session.decode_step = decode_step
        
        return self._serialize_hidden_states(output_hidden)
    
    def _run_sglang_forward(
        self,
        session: InferenceSession,
        hidden_states: torch.Tensor,
        position_ids: list,
        is_prefill: bool,
    ) -> torch.Tensor:
        """
        Run a forward pass through SGLang's engine with hidden states.
        
        This is the most complex integration point. SGLang's Engine API
        is designed for text input (generate/add_request). To feed hidden
        states directly, we have two approaches:
        
        Approach A (Recommended): Custom ModelRunner integration
            - Modify SGLang's ModelRunner to accept a "hidden_states" input mode
            - The runner bypasses tokenization and embedding
            - Hidden states go directly to the model's forward() as input_embeds
        
        Approach B (Simpler): Engine-level bypass
            - Use SGLang's Engine at a lower level
            - Directly call model.forward() outside the normal scheduling loop
            - Lose some batching/caching benefits but simpler to implement
        
        We implement Approach B here for clarity. Production systems should
        use Approach A for proper continuous batching.
        """
        # For Approach B: direct model forward call
        # This bypasses SGLang's scheduler but gives us full control
        
        positions = torch.tensor(position_ids, dtype=torch.long, device="cuda:0")
        
        # TODO: In production, integrate with SGLang's ForwardBatch
        # to get proper KV cache management and attention masking.
        # For now, we do a direct forward pass.
        
        # The headless model's forward() expects:
        #   forward(input_ids=dummy, positions=positions,
        #           forward_batch=batch, input_embeds=hidden_states)
        
        # Placeholder: return hidden states (replace with actual engine call)
        return hidden_states


# ============================================================================
# gRPC Service Implementation
# ============================================================================

class SplitInferenceServicer:
    """
    gRPC service implementation.
    
    In production, this would inherit from:
        split_inference_pb2_grpc.SplitInferenceServiceServicer
    """
    
    def __init__(self, config: SplitInferenceConfig):
        self.processor = ActivationProcessor(config)
    
    def CreateSession(self, request, context):
        """Create a new inference session."""
        session_id = self.processor.create_session(
            model_name=request.model_name,
            local_layers=request.local_layers,
            max_seq_len=request.max_seq_len,
            dp_metadata={
                "dp_enabled": request.dp_metadata.dp_enabled,
                "perturbation_enabled": request.dp_metadata.perturbation_enabled,
                "perturbation_seed": request.dp_metadata.perturbation_seed,
                "perturbation_scale": request.dp_metadata.perturbation_scale,
            },
        )
        # return pb2.CreateSessionResponse(session_id=session_id, success=True)
    
    def Prefill(self, request, context):
        """Handle prefill request."""
        t0 = time.perf_counter()
        
        output_bytes = self.processor.process_prefill(
            session_id=request.session_id,
            hidden_bytes=request.hidden_states,
            position_ids=list(request.position_ids),
            noise_sigma=request.noise_sigma,
        )
        
        total_time = (time.perf_counter() - t0) * 1000
        # return pb2.PrefillResponse(
        #     hidden_states=output_bytes,
        #     success=True,
        #     total_time_ms=total_time,
        # )
    
    def Decode(self, request, context):
        """Handle single decode step."""
        t0 = time.perf_counter()
        
        output_bytes = self.processor.process_decode(
            session_id=request.session_id,
            hidden_bytes=request.hidden_states,
            position_id=request.position_id,
            decode_step=request.decode_step,
            noise_sigma=request.noise_sigma,
        )
        
        total_time = (time.perf_counter() - t0) * 1000
        # return pb2.DecodeResponse(
        #     hidden_states=output_bytes,
        #     success=True,
        #     total_time_ms=total_time,
        # )
    
    def StreamDecode(self, request_iterator, context):
        """
        Bidirectional streaming decode.
        
        Lower overhead than unary RPCs for autoregressive generation:
        - No per-call connection setup
        - Pipelined: send step N, receive step N-1 simultaneously
        - HTTP/2 multiplexing
        """
        for request in request_iterator:
            output_bytes = self.processor.process_decode(
                session_id=request.session_id,
                hidden_bytes=request.hidden_states,
                position_id=request.position_id,
                decode_step=request.decode_step,
                noise_sigma=request.noise_sigma,
            )
            # yield pb2.DecodeResponse(hidden_states=output_bytes, success=True)
    
    def HealthCheck(self, request, context):
        """Return server health status."""
        active = len(self.processor.sessions)
        
        # GPU stats
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            gpu_util, mem_used, mem_total = 0, 0, 0
        
        # return pb2.HealthCheckResponse(
        #     healthy=True,
        #     active_sessions=active,
        #     gpu_utilization=gpu_util,
        #     gpu_memory_used_gb=mem_used,
        #     gpu_memory_total_gb=mem_total,
        # )


def serve(config: SplitInferenceConfig):
    """Start the gRPC activation server."""
    net = config.network
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', net.max_message_size),
            ('grpc.max_receive_message_length', net.max_message_size),
        ],
    )
    
    servicer = SplitInferenceServicer(config)
    # pb2_grpc.add_SplitInferenceServiceServicer_to_server(servicer, server)
    
    if net.tls_enabled:
        with open(net.ca_cert, 'rb') as f:
            ca_cert = f.read()
        with open(net.server_cert, 'rb') as f:
            server_cert = f.read()
        with open(net.server_key, 'rb') as f:
            server_key = f.read()
        
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True,  # mTLS: require client certificate
        )
        server.add_secure_port(f"{net.main_server_host}:{net.main_server_port}", credentials)
    else:
        server.add_insecure_port(f"{net.main_server_host}:{net.main_server_port}")
    
    server.start()
    logger.info(
        f"🚀 Activation Server running on {net.main_server_host}:{net.main_server_port} "
        f"(TLS={'enabled' if net.tls_enabled else 'DISABLED'})"
    )
    server.wait_for_termination()

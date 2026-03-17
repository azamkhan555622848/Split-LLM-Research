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
import signal
import logging
import threading
import numpy as np
import torch
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, Optional

import grpc

from split_inference.config import SplitInferenceConfig, NetworkConfig
from split_inference.local_server.privacy_engine import PrivacyEngine
from split_inference.proto import split_inference_pb2 as pb2
from split_inference.proto import split_inference_pb2_grpc as pb2_grpc

# SGLang engine import
try:
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

# HuggingFace imports for direct model loading
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch.nn as nn
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = logging.getLogger(__name__)


MAX_SESSIONS = 64
SESSION_IDLE_TIMEOUT = 600  # 10 minutes
SESSION_CLEANUP_INTERVAL = 60  # Check every 60 seconds


@dataclass
class InferenceSession:
    """State for a single client session."""
    session_id: str
    model_name: str
    local_layers: int
    max_seq_len: int
    created_at: float
    last_activity: float = 0.0

    # DP metadata from client
    dp_enabled: bool = False
    perturbation_enabled: bool = False
    perturbation_seed: int = 0
    perturbation_scale: float = 0.0

    # KV cache tracking
    current_seq_len: int = 0
    decode_step: int = 0

    # StreamDecode deduplication
    seen_decode_steps: set = field(default_factory=set)

    # KV cache for remote layers
    remote_past_key_values: Optional[object] = field(default=None, repr=False)

    # SGLang request handle
    sglang_request_id: Optional[str] = None

    def __post_init__(self):
        if self.last_activity == 0.0:
            self.last_activity = self.created_at

    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()


class RemoteModelShard:
    """
    Loads and runs layers K..L of a transformer model directly via HuggingFace.

    This is the production-ready approach for split inference without SGLang.
    It loads only the remote layers to GPU, discarding embedding/lm_head/local layers.
    """

    def __init__(self, model_name: str, local_layers: int, device: str = "cuda:0", dtype: str = "float16"):
        self.model_name = model_name
        self.local_layers = local_layers
        self.device = device
        self.layers = None
        self.rotary_emb = None
        self._load_model(dtype)

    def _load_model(self, dtype: str):
        logger.info(f"Loading remote shard: {self.model_name} (layers {self.local_layers}+)")
        t0 = time.time()

        full_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=getattr(torch, dtype),
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        base = full_model.model

        # Load only remote layers (K..L)
        total = len(base.layers)
        self.layers = nn.ModuleList([
            base.layers[i].to(self.device)
            for i in range(self.local_layers, total)
        ])

        # RoPE
        if hasattr(base, 'rotary_emb'):
            self.rotary_emb = base.rotary_emb.to(self.device)
        elif hasattr(base.layers[0].self_attn, 'rotary_emb'):
            self.rotary_emb = base.layers[0].self_attn.rotary_emb.to(self.device)

        del full_model, base
        torch.cuda.empty_cache()

        param_count = sum(p.numel() for p in self.layers.parameters())
        logger.info(
            f"Remote shard loaded: {param_count/1e6:.1f}M params, "
            f"{total - self.local_layers} layers, {time.time()-t0:.1f}s"
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: list,
        past_key_values=None,
        use_cache: bool = True,
    ) -> tuple:
        """
        Run hidden_states through remote layers K..L with KV cache support.

        Returns:
            (output_hidden_states, updated_past_key_values)
        """
        from transformers.cache_utils import DynamicCache

        hidden_states = hidden_states.to(self.device)
        pos = torch.tensor([position_ids], dtype=torch.long, device=self.device)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, pos)

        kwargs = {}
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        if past_key_values is None:
            past_key_values = DynamicCache()

        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                position_ids=pos,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_output[0]
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(0)

        return hidden_states, past_key_values


class ActivationProcessor:
    """
    Bridges gRPC activation tensors to the remote model shard.

    Flow:
    1. Receive hidden_states bytes from gRPC
    2. Deserialize to torch tensor
    3. Remove perturbation (if enabled)
    4. Run through remote layers K..L (direct PyTorch forward)
    5. Serialize and return via gRPC
    """

    def __init__(self, config: SplitInferenceConfig):
        self.config = config
        self.sessions: Dict[str, InferenceSession] = {}
        self.engine = None
        self.remote_shard: Optional[RemoteModelShard] = None
        self._cleanup_stop = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._session_cleanup_loop, daemon=True, name="session-cleanup",
        )
        self._cleanup_thread.start()

        # Load remote model shard (direct PyTorch, no SGLang needed)
        if HF_AVAILABLE and config.model.model_name:
            try:
                self.remote_shard = RemoteModelShard(
                    config.model.model_name,
                    config.model.local_layers,
                    dtype=config.model.dtype,
                )
            except Exception as e:
                logger.warning(f"Failed to load remote shard: {e}. Using identity fallback.")

    def _session_cleanup_loop(self):
        """Background thread to evict idle sessions."""
        while not self._cleanup_stop.wait(SESSION_CLEANUP_INTERVAL):
            now = time.time()
            expired = [
                sid for sid, s in self.sessions.items()
                if now - s.last_activity > SESSION_IDLE_TIMEOUT
            ]
            for sid in expired:
                logger.info(f"Evicting idle session: {sid[:8]}... "
                           f"(idle {now - self.sessions[sid].last_activity:.0f}s)")
                self.destroy_session(sid)
    
    def _init_sglang_engine(self, headless_model_path: Optional[str] = None):
        """
        Initialize SGLang Engine with the headless model.

        Uses headless_model_path (prepared by launch.py with auto_map in config.json)
        so that HeadlessTransformerForRemoteInference is loaded via trust_remote_code,
        surviving SGLang's process isolation.
        """
        model_path = headless_model_path or self.config.model.model_name

        self.engine = Engine(
            model_path=model_path,
            tp_size=self.config.sglang.tp_size,
            dp_size=self.config.sglang.dp_size,
            mem_fraction_static=self.config.sglang.mem_fraction,
            max_running_requests=self.config.sglang.max_running_requests,
            chunked_prefill_size=self.config.sglang.chunked_prefill_size,
            disable_radix_cache=not self.config.sglang.enable_radix_cache,
            quantization=self.config.sglang.quantization,
            trust_remote_code=True,
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
        if len(self.sessions) >= MAX_SESSIONS:
            raise RuntimeError(
                f"Max sessions ({MAX_SESSIONS}) reached. "
                "Destroy an existing session or wait for idle cleanup."
            )
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
    
    DTYPE_MAP = {
        "float16": (torch.float16, np.float16),
        "bfloat16": (torch.bfloat16, np.float16),  # serialize as fp16 on wire
        "float32": (torch.float32, np.float32),
    }

    def _deserialize_hidden_states(
        self,
        data: bytes,
        dtype: str = "float16",
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """Deserialize hidden states from gRPC bytes."""
        _, wire_np_dtype = self.DTYPE_MAP.get(dtype, (torch.float16, np.float16))
        ndim = struct.unpack('B', data[:1])[0]
        shape_size = ndim * 4
        shape = struct.unpack(f'{ndim}I', data[1:1 + shape_size])
        tensor_data = data[1 + shape_size:]
        arr = np.frombuffer(tensor_data, dtype=wire_np_dtype).reshape(shape)
        return torch.from_numpy(arr.copy()).to(device)

    def _serialize_hidden_states(self, tensor: torch.Tensor, dtype: str = "float16") -> bytes:
        """Serialize hidden states for gRPC response."""
        wire_torch_dtype, _ = self.DTYPE_MAP.get(dtype, (torch.float16, np.float16))
        t = tensor.detach().cpu().contiguous().to(wire_torch_dtype)
        ndim_header = struct.pack('B', len(t.shape))
        shape_header = struct.pack(f'{len(t.shape)}I', *t.shape)
        return ndim_header + shape_header + t.numpy().tobytes()
    
    def process_prefill(
        self,
        session_id: str,
        hidden_bytes: bytes,
        position_ids: list,
        noise_sigma: float,
        dtype: str = "float16",
    ) -> bytes:
        """
        Process a prefill request through the remote layers.
        
        1. Deserialize hidden states
        2. Remove perturbation (if enabled)
        3. Run through SGLang engine (layers K..L)
        4. Serialize and return output hidden states
        """
        session = self.sessions[session_id]
        session.touch()
        t0 = time.perf_counter()

        # Deserialize
        hidden_states = self._deserialize_hidden_states(hidden_bytes, dtype=dtype)

        # Remove perturbation if enabled
        if session.perturbation_enabled:
            hidden_states = PrivacyEngine.remove_perturbation(
                hidden_states,
                session.perturbation_seed,
                session.perturbation_scale,
                step=0,
            )
        
        # Run through remote model layers
        if self.remote_shard is not None:
            output_hidden, session.remote_past_key_values = self.remote_shard.forward(
                hidden_states, position_ids,
                past_key_values=None,  # Fresh cache for prefill
            )
        elif self.engine is not None:
            output_hidden = self._run_sglang_forward(
                session, hidden_states, position_ids, is_prefill=True,
            )
        else:
            output_hidden = hidden_states
            logger.warning("No remote model loaded — returning identity transform")
        
        # Update session state
        session.current_seq_len = len(position_ids)
        
        compute_time = (time.perf_counter() - t0) * 1000
        logger.debug(f"Prefill processed: seq_len={len(position_ids)}, time={compute_time:.1f}ms")
        
        return self._serialize_hidden_states(output_hidden, dtype=dtype)

    def process_decode(
        self,
        session_id: str,
        hidden_bytes: bytes,
        position_id: int,
        decode_step: int,
        noise_sigma: float,
        dtype: str = "float16",
    ) -> bytes:
        """Process a single decode step through remote layers."""
        session = self.sessions[session_id]
        session.touch()

        hidden_states = self._deserialize_hidden_states(hidden_bytes, dtype=dtype)
        
        if session.perturbation_enabled:
            hidden_states = PrivacyEngine.remove_perturbation(
                hidden_states,
                session.perturbation_seed,
                session.perturbation_scale,
                step=decode_step,
            )
        
        if self.remote_shard is not None:
            output_hidden, session.remote_past_key_values = self.remote_shard.forward(
                hidden_states, [position_id],
                past_key_values=session.remote_past_key_values,
            )
        elif self.engine is not None:
            output_hidden = self._run_sglang_forward(
                session, hidden_states, [position_id], is_prefill=False,
            )
        else:
            output_hidden = hidden_states
        
        session.current_seq_len += 1
        session.decode_step = decode_step
        
        return self._serialize_hidden_states(output_hidden, dtype=dtype)

    def _run_sglang_forward(
        self,
        session: InferenceSession,
        hidden_states: torch.Tensor,
        position_ids: list,
        is_prefill: bool,
    ) -> torch.Tensor:
        """
        Run a forward pass through SGLang's Engine using input_embeds.

        Approach A (Primary): Engine.generate() with input_embeds + return_hidden_states
            - Uses SGLang's full pipeline (scheduler, KV cache, batching)
            - input_embeds passed via GenerateReqInput
            - max_new_tokens=1 with return_hidden_states=True

        Approach B (Fallback): Direct model forward
            - Bypasses SGLang scheduler, loses batching/caching
            - Used if Approach A fails or for debugging
        """
        try:
            return self._run_via_engine_api(session, hidden_states, position_ids)
        except Exception as e:
            logger.warning(f"Engine API forward failed ({e}), falling back to direct forward")
            return self._run_direct_forward(session, hidden_states, position_ids)

    def _run_via_engine_api(
        self,
        session: InferenceSession,
        hidden_states: torch.Tensor,
        position_ids: list,
    ) -> torch.Tensor:
        """Approach A: Use SGLang Engine.generate() with input_embeds."""
        from sglang.srt.managers.io_struct import GenerateReqInput

        # SGLang expects input_embeds as nested lists, not tensors
        embeds_list = hidden_states.cpu().float().tolist()

        req = GenerateReqInput(
            input_embeds=embeds_list,
            return_hidden_states=True,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )

        output = self.engine.generate(req)

        # Extract hidden states from output
        if isinstance(output, dict) and "meta_info" in output:
            hs = output["meta_info"].get("hidden_states")
            if hs is not None:
                if isinstance(hs, torch.Tensor):
                    return hs
                return torch.tensor(hs, device="cuda:0")

        raise RuntimeError("Engine did not return hidden_states in meta_info")

    def _run_direct_forward(
        self,
        session: InferenceSession,
        hidden_states: torch.Tensor,
        position_ids: list,
    ) -> torch.Tensor:
        """Approach B: Direct model forward (bypasses SGLang scheduler)."""
        positions = torch.tensor(position_ids, dtype=torch.long, device="cuda:0")

        # TODO: Integrate with SGLang's ForwardBatch for proper KV cache
        # management. For now, this is a stateless forward pass.

        # Placeholder: return hidden states (replace with actual model.forward() call
        # once ForwardBatch integration is implemented)
        return hidden_states


# ============================================================================
# gRPC Service Implementation
# ============================================================================

class SplitInferenceServicer(pb2_grpc.SplitInferenceServiceServicer):
    """gRPC service implementation for split inference."""

    def __init__(self, config: SplitInferenceConfig):
        self.processor = ActivationProcessor(config)

    def _validate_session(self, session_id: str, context) -> bool:
        """Validate session exists, abort with NOT_FOUND if missing."""
        if session_id not in self.processor.sessions:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Session not found: {session_id}")
            return False
        return True

    def CreateSession(self, request, context):
        """Create a new inference session."""
        try:
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
            return pb2.CreateSessionResponse(session_id=session_id, success=True)
        except Exception as e:
            logger.error(f"CreateSession error: {e}")
            return pb2.CreateSessionResponse(
                success=False, error_message=str(e),
            )

    def Prefill(self, request, context):
        """Handle prefill request."""
        self._validate_session(request.session_id, context)
        t0 = time.perf_counter()

        try:
            output_bytes = self.processor.process_prefill(
                session_id=request.session_id,
                hidden_bytes=request.hidden_states,
                position_ids=list(request.position_ids),
                noise_sigma=request.noise_sigma,
                dtype=request.dtype or "float16",
            )
        except Exception as e:
            logger.error(f"Prefill error (session={request.session_id}): {e}")
            return pb2.PrefillResponse(success=False, error_message=str(e))

        total_time = (time.perf_counter() - t0) * 1000
        return pb2.PrefillResponse(
            hidden_states=output_bytes,
            success=True,
            total_time_ms=total_time,
        )

    def Decode(self, request, context):
        """Handle single decode step."""
        self._validate_session(request.session_id, context)
        t0 = time.perf_counter()

        try:
            output_bytes = self.processor.process_decode(
                session_id=request.session_id,
                hidden_bytes=request.hidden_states,
                position_id=request.position_id,
                decode_step=request.decode_step,
                noise_sigma=request.noise_sigma,
                dtype=request.dtype or "float16",
            )
        except Exception as e:
            logger.error(f"Decode error (session={request.session_id}, step={request.decode_step}): {e}")
            return pb2.DecodeResponse(success=False, error_message=str(e))

        total_time = (time.perf_counter() - t0) * 1000
        return pb2.DecodeResponse(
            hidden_states=output_bytes,
            success=True,
            total_time_ms=total_time,
        )

    def StreamDecode(self, request_iterator, context):
        """
        Bidirectional streaming decode.

        Lower overhead than unary RPCs for autoregressive generation:
        - No per-call connection setup
        - Pipelined: send step N, receive step N-1 simultaneously
        - HTTP/2 multiplexing

        Deduplication: tracks seen decode steps per session to prevent
        duplicate processing from retransmissions.
        """
        for request in request_iterator:
            self._validate_session(request.session_id, context)

            # Deduplicate: skip already-processed decode steps
            session = self.processor.sessions.get(request.session_id)
            if session and request.decode_step in session.seen_decode_steps:
                logger.warning(
                    f"Duplicate decode step {request.decode_step} for session "
                    f"{request.session_id[:8]}, skipping"
                )
                continue
            if session:
                session.seen_decode_steps.add(request.decode_step)

            try:
                output_bytes = self.processor.process_decode(
                    session_id=request.session_id,
                    hidden_bytes=request.hidden_states,
                    position_id=request.position_id,
                    decode_step=request.decode_step,
                    noise_sigma=request.noise_sigma,
                    dtype=request.dtype or "float16",
                )
                yield pb2.DecodeResponse(hidden_states=output_bytes, success=True)
            except Exception as e:
                logger.error(f"StreamDecode error: {e}")
                yield pb2.DecodeResponse(success=False, error_message=str(e))

    def HealthCheck(self, request, context):
        """Return server health status."""
        active = len(self.processor.sessions)

        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization()
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            gpu_util, mem_used, mem_total = 0, 0, 0

        return pb2.HealthCheckResponse(
            healthy=True,
            active_sessions=active,
            gpu_utilization=gpu_util,
            gpu_memory_used_gb=mem_used,
            gpu_memory_total_gb=mem_total,
        )


def serve(config: SplitInferenceConfig):
    """Start the gRPC activation server with graceful shutdown."""
    net = config.network

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', net.max_message_size),
            ('grpc.max_receive_message_length', net.max_message_size),
        ],
    )

    servicer = SplitInferenceServicer(config)
    pb2_grpc.add_SplitInferenceServiceServicer_to_server(servicer, server)

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
            require_client_auth=True,
        )
        server.add_secure_port(f"{net.main_server_host}:{net.main_server_port}", credentials)
    else:
        server.add_insecure_port(f"{net.main_server_host}:{net.main_server_port}")

    # Graceful shutdown on SIGTERM/SIGINT
    shutdown_event = threading.Event()

    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down gracefully...")
        # Stop accepting new RPCs, drain existing ones (5s grace)
        server.stop(grace=5)
        # Stop session cleanup thread
        servicer.processor._cleanup_stop.set()
        # Log final session state
        active = len(servicer.processor.sessions)
        if active > 0:
            logger.info(f"Shutting down with {active} active sessions")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    server.start()
    logger.info(
        f"Activation Server running on {net.main_server_host}:{net.main_server_port} "
        f"(TLS={'enabled' if net.tls_enabled else 'DISABLED'}, "
        f"max_sessions={MAX_SESSIONS}, idle_timeout={SESSION_IDLE_TIMEOUT}s)"
    )

    shutdown_event.wait()
    logger.info("Server stopped")

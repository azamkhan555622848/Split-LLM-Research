# Split LLM Inference — Project Guide

## Overview
Privacy-preserving split LLM inference system. Raw tokens/text/logits never leave the local machine — only DP-noised activations over mTLS gRPC.

## Architecture
- **Local Server** (`split_inference/local_server/`): Embedding + first K layers + LM head + privacy engine. All user data stays here.
- **Main Server** (`split_inference/main_server/`): Layers K..L via SGLang headless model. Only sees noisy activations.
- **Communication**: gRPC with protobuf serialization (float16 bytes), mTLS transport encryption.
- **Privacy**: DP noise (Gaussian/Laplace), activation clipping, optional reversible perturbation, RDP accounting.

## Hardware Target
3x RTX A6000 (144GB VRAM) — Llama 8B (TP=1), 70B (TP=3)

## Package Structure
```
split_inference/
├── __init__.py
├── config.py                  # Shared dataclass configs (ModelConfig, PrivacyConfig, etc.)
├── proto/
│   ├── __init__.py
│   ├── split_inference.proto   # gRPC service definition
│   ├── split_inference_pb2.py  # GENERATED — do not edit
│   ├── split_inference_pb2_grpc.py
│   └── split_inference_pb2.pyi
├── local_server/
│   ├── __init__.py
│   ├── client.py              # Main client orchestrating split inference loop
│   ├── local_model.py         # LocalModelShard — embedding + K layers + LM head
│   └── privacy_engine.py      # DP noise, clipping, perturbation, RDP accounting
├── main_server/
│   ├── __init__.py
│   ├── activation_server.py   # gRPC service + ActivationProcessor
│   ├── headless_llama.py      # Custom SGLang model (layers K..L only)
│   └── launch.py              # Server startup orchestration
├── crypto/
│   ├── __init__.py
│   └── channel.py             # mTLS cert generation, TEE attestation stubs
scripts/
├── build_proto.py             # Compile .proto → pb2/pb2_grpc
```

## Key Commands
```bash
pip install -e ".[server,dev]"       # Install package
python scripts/build_proto.py        # Compile protobuf (must run after proto changes)
split-gencerts                       # Generate mTLS certificates
split-server --model <name> --tp 1   # Launch main server
split-local --model <name>           # Launch local client
```

## Implementation Status

### Phase 1: Foundation (COMPLETED)
- [x] `pyproject.toml` — package with deps, optional groups (`server`, `dev`), CLI entry points (`split-local`, `split-server`, `split-gencerts`)
- [x] `__init__.py` files in all 5 subpackages
- [x] All imports fixed — removed `sys.path.append` hacks in 5 files, using absolute `split_inference.*` imports
- [x] `scripts/build_proto.py` — compiles proto with automatic relative import fix in generated grpc file
- [x] `.gitignore` — excludes generated `*_pb2*.py`, `certs/`, `__pycache__/`, `.venv/`, `*.egg-info/`
- [x] Protobuf compiled successfully — `split_inference_pb2.py`, `_pb2_grpc.py`, `_pb2.pyi` generated

### Phase 2: Activate gRPC — Real Client-Server Communication (PENDING)
- [ ] Uncomment client gRPC in `local_server/client.py`:
  - Lines 31-32: `from split_inference.proto import split_inference_pb2 as pb2, split_inference_pb2_grpc as pb2_grpc`
  - Line 146: `self.stub = pb2_grpc.SplitInferenceServiceStub(self.channel)`
  - Lines 151-169: Full `CreateSession` RPC, remove `pass` placeholder
  - Lines 203-216: `PrefillRequest` + `stub.Prefill()`, remove mock `remote_hidden = protected_hidden`
  - Lines 258-270: `DecodeRequest` + `stub.Decode()`, remove mock
- [ ] Uncomment server gRPC in `main_server/activation_server.py`:
  - Make `SplitInferenceServicer` inherit `pb2_grpc.SplitInferenceServiceServicer`
  - Uncomment all `return pb2.*Response(...)` in CreateSession, Prefill, Decode, StreamDecode, HealthCheck
  - Uncomment `pb2_grpc.add_SplitInferenceServiceServicer_to_server(servicer, server)`
- [ ] Add error handling: `try/except grpc.RpcError` with logging, timeouts (30s prefill, 10s decode)
- [ ] Validate session_id in Prefill/Decode, `context.abort(NOT_FOUND)` for missing sessions
- [ ] Fix `ActivationSerializer` dtype handling — accept dtype param instead of hardcoding float16

### Phase 3: SGLang Integration via Engine `input_embeds` (PENDING — Most Complex)
- [ ] Refactor `HeadlessLlamaModel` → Generic `HeadlessTransformerModel` in `main_server/headless_llama.py`:
  - Architecture detection from HF `config.json` `"architectures"` field
  - `SUPPORTED_ARCHITECTURES` dict mapping layer accessor paths per arch (Llama, Mistral, Qwen2 share same paths)
  - Register as `"HeadlessModelForRemoteInference"` in ModelRegistry
- [ ] Implement Engine-level `input_embeds` integration in `main_server/activation_server.py`:
  - Replace `_run_sglang_forward()` with `GenerateReqInput(input_embeds=..., return_hidden_states=True, max_new_tokens=0)`
  - Handle KV cache continuity: map `session_id` → SGLang request ID
  - Fallback to direct model forward (Approach B) if `input_embeds + return_hidden_states` doesn't work
- [ ] Update `_init_sglang_engine()`: register generic model, support `tp_size=1` (8B) and `tp_size=3` (70B)
- [ ] Update `launch.py`: detect original architecture in `prepare_headless_config()`, add `--tp` support
- [ ] Update `LocalModelShard` in `local_server/local_model.py`:
  - `ARCH_COMPONENTS` dict for component extraction paths per architecture
  - Auto-detect architecture from `AutoConfig`
- [ ] Add `ModelConfig.from_pretrained()` in `config.py`:
  - Auto-detect `total_layers`, `hidden_dim`, `num_heads`, `head_dim`, `vocab_size` from HF config

### Phase 4: Privacy Fixes — Correct DP Guarantees (PENDING — Can Parallel with Phase 3)
- [ ] **CRITICAL BUG**: Fix prefill per-token DP accounting in `local_server/privacy_engine.py:add_dp_noise()`:
  - Currently: prefill of N tokens counts as 1 DP step → epsilon severely underestimated
  - Fix: `num_tokens = hidden_states.shape[-2]`, loop `accountant.step()` N times
- [ ] Fix sensitivity estimation in `estimate_activation_sensitivity()`:
  - Change 95th → 99.9th percentile with 1.5x safety margin
  - Add `sensitivity_mode` config: `"empirical"` vs `"clip_norm"` (conservative default)
  - Fix broken `model.forward_to_layer()` → `model.forward_to_split()`
- [ ] Key file permissions in `crypto/channel.py`: add `os.chmod(key_path, 0o600)` after key generation
- [ ] StreamDecode deduplication in `main_server/activation_server.py`: track `seen_steps: Set[int]` per session

### Phase 5: Production Hardening (PENDING — Depends on Phases 2-4)
- [ ] Session management in `activation_server.py`: `last_activity` tracking, `max_sessions=64`, background cleanup (evict >10 min idle)
- [ ] Graceful shutdown: SIGTERM/SIGINT → `server.stop(grace=5)`, drain RPCs, flush privacy logs
- [ ] Client resilience in `client.py`: retry with exponential backoff (3 attempts, 1s/2s/4s), `close()` + context manager, channel connectivity monitoring
- [ ] CLI configuration in `client.py:main()`: argparse with `--model`, `--local-layers`, `--server-address`, `--dp-epsilon`, `--tls-enabled`; env var overrides `SPLIT_INFERENCE_*`
- [ ] Config validation in `config.py`: assert valid ranges for all parameters
- [ ] Deployment files: `Dockerfile.local`, `Dockerfile.server`, `docker-compose.yml` (GPU reservation)
- [ ] Logging: replace `print()` → `logger.info()`, structured JSON logging option, session_id correlation

### Phase 6: Testing (PENDING — Can Start After Phase 2)
- [ ] `tests/test_serializer.py` — round-trip shapes `[1,4096]`, `[128,4096]`, `[1,1,4096]`, dtype handling
- [ ] `tests/test_privacy_engine.py` — clipping correctness, noise calibration, **N-token prefill → N steps**, perturbation reversibility, RDP composition
- [ ] `tests/test_grpc_integration.py` — full lifecycle: CreateSession → Prefill → Decode → Destroy → HealthCheck, error cases
- [ ] `tests/test_headless_model.py` — weight loading (local layers skipped, indices remapped), multi-arch
- [ ] `tests/test_multi_arch.py` — `ModelConfig.from_pretrained()` auto-detection, `LocalModelShard` per architecture

## Verification Plan

### End-to-End Smoke Test
```bash
pip install -e ".[server,dev]"
python scripts/build_proto.py
split-gencerts
# Terminal 1:
split-server --model meta-llama/Llama-3.1-8B-Instruct --local-layers 2 --tp 1 --tls-enabled
# Terminal 2:
split-local --model meta-llama/Llama-3.1-8B-Instruct --server-address localhost:50051 --dp-epsilon 8.0 --tls-enabled
```

### Privacy Verification
- Prefill 100 tokens → accountant shows 100 steps (not 1)
- Compare output DP-on vs DP-off → noise present
- Perturbation round-trip within float16 precision

### Multi-Architecture Test
```bash
split-server --model mistralai/Mistral-7B-Instruct-v0.3 --local-layers 2 --tp 1
split-server --model Qwen/Qwen2-7B-Instruct --local-layers 2 --tp 1
```

### Performance Baseline
- Measure prefill + decode latency decomposition
- Compare tok/s against arxiv:2602.16760 benchmarks (8-19 tok/s)

## Conventions
- All imports use absolute paths: `from split_inference.config import ...`
- Proto files regenerated via `scripts/build_proto.py` (fixes relative imports automatically)
- SGLang imports wrapped in `try/except ImportError` with `SGLANG_AVAILABLE` flag
- Config uses dataclasses in `config.py`, not separate config files

## Key Research References
- arxiv:2602.16760: RTT dominates (64%), speculative decoding → 1.2-1.3 tok/step, 15-19 tok/s at 20ms RTT
- SGLang `GenerateReqInput` has `input_embeds` field (GitHub #745, DeepWiki docs)
- SGLang supports `return_hidden_states` via `output["meta_info"]["hidden_states"]`
- Custom model registration: `ModelRegistry.models[name] = class` (process isolation caveat: #11578)
- Balle & Wang 2018: Analytic Gaussian Mechanism for DP noise calibration

## Known Bugs (To Fix in Phase 4)
- **Critical**: `privacy_engine.py:add_dp_noise()` counts prefill of N tokens as 1 DP step — epsilon severely underestimated
- `estimate_activation_sensitivity()` calls `model.forward_to_layer()` which doesn't exist — should be `model.forward_to_split()`
- Sensitivity estimation uses 95th percentile — should use 99.9th with 1.5x safety margin

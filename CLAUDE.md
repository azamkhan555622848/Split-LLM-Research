# Split LLM Inference — Project Guide

## Overview
Privacy-preserving split LLM inference system. Raw tokens/text/logits never leave the local machine — only DP-noised activations over mTLS gRPC.

## Architecture
- **Local Server** (`split_inference/local_server/`): Embedding + first K layers + LM head + privacy engine. All user data stays here.
- **Main Server** (`split_inference/main_server/`): Layers K..L via `RemoteModelShard` (direct PyTorch) or SGLang headless model. Only sees noisy activations.
- **Communication**: gRPC with protobuf serialization (float16 bytes), mTLS transport encryption.
- **Privacy**: DP noise (Gaussian/Laplace), activation clipping, optional reversible perturbation, RDP accounting.

## Hardware — Verified Working
- **Local**: RTX 4070 Super (12GB VRAM, Windows 11) — runs local shard (~3.1 GB for Qwen2-7B)
- **Remote**: 3x RTX A6000 (49GB each, Ubuntu 24.04) at `asus@140.113.202.36` — runs remote shard (~12 GB for Qwen2-7B)
- **Verified E2E**: Qwen2-7B-Instruct split at layer 2, cross-machine gRPC, 12.8 tok/s cached

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
│   ├── activation_server.py   # gRPC service + ActivationProcessor + RemoteModelShard
│   ├── headless_llama.py      # HeadlessTransformerForRemoteInference (multi-arch SGLang model)
│   └── launch.py              # Server startup orchestration
├── crypto/
│   ├── __init__.py
│   └── channel.py             # mTLS cert generation, TEE attestation stubs
scripts/
├── build_proto.py             # Compile .proto → pb2/pb2_grpc
├── start_server.sh            # Launch remote server on GPU machine
tests/
├── test_serializer.py         # 8 tests — round-trip shapes, dtypes
├── test_privacy_engine.py     # 20 tests — clipping, noise, accounting, perturbation
├── test_grpc_integration.py   # 11 tests — full gRPC lifecycle
├── test_headless_model.py     # 8 tests — multi-arch, weight loading
├── test_multi_arch.py         # 13 tests — config validation, auto-detection
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

### Phase 2: Activate gRPC — Real Client-Server Communication (COMPLETED)
- [x] Uncomment client gRPC in `local_server/client.py`:
  - Proto imports activated: `from split_inference.proto import split_inference_pb2 as pb2, split_inference_pb2_grpc as pb2_grpc`
  - Stub creation: `self.stub = pb2_grpc.SplitInferenceServiceStub(self.channel)`
  - Full `CreateSession` RPC with error handling, removed `pass` placeholder
  - `PrefillRequest` + `stub.Prefill()` with 30s timeout, removed mock
  - `DecodeRequest` + `stub.Decode()` with 10s timeout, removed mock
- [x] Uncomment server gRPC in `main_server/activation_server.py`:
  - `SplitInferenceServicer` inherits `pb2_grpc.SplitInferenceServiceServicer`
  - All `return pb2.*Response(...)` activated in CreateSession, Prefill, Decode, StreamDecode, HealthCheck
  - `pb2_grpc.add_SplitInferenceServiceServicer_to_server(servicer, server)` activated
- [x] Add error handling: `try/except grpc.RpcError` on client, `try/except Exception` on server with logging, timeouts (30s prefill/session, 10s decode)
- [x] Validate session_id in Prefill/Decode/StreamDecode via `_validate_session()` → `context.abort(NOT_FOUND)`
- [x] Fix `ActivationSerializer` dtype handling — both client and server accept `dtype` param with `DTYPE_MAP` supporting float16/bfloat16/float32

### Phase 3: SGLang Integration via Engine `input_embeds` (COMPLETED)
- [x] Refactor `HeadlessLlamaModel` → `HeadlessTransformerModel` in `main_server/headless_llama.py`:
  - `ARCH_REGISTRY` maps architectures to decoder layer classes (Llama, Mistral → LlamaDecoderLayer; Qwen2 → Qwen2DecoderLayer)
  - `_get_decoder_layer_class()` dynamically imports correct layer class per architecture
  - `HeadlessTransformerForRemoteInference` reads `original_architecture` from config for dispatch
  - Backwards compat alias: `HeadlessLlamaForRemoteInference = HeadlessTransformerForRemoteInference`
- [x] Implement Engine-level `input_embeds` integration in `main_server/activation_server.py`:
  - `_run_via_engine_api()`: `GenerateReqInput(input_embeds=..., return_hidden_states=True, max_new_tokens=1)`
  - `_run_direct_forward()`: Approach B fallback for when Engine API fails
  - `_run_sglang_forward()` tries Approach A first, falls back to B
- [x] Update `_init_sglang_engine()`: uses `trust_remote_code=True`, fixed `max_running_requests` (was `max_num_reqs`), accepts `headless_model_path`
- [x] Update `launch.py`:
  - `prepare_headless_config()` detects `original_architecture`, writes `auto_map` + `headless_model.py` for process-isolation-safe loading
  - `_write_headless_model_module()` generates re-export module for `trust_remote_code`
  - All imports fixed to absolute `split_inference.*` paths
  - `main()` uses `ModelConfig.from_pretrained()` for auto-detection
- [x] Update `LocalModelShard` in `local_server/local_model.py`:
  - Auto-detects architecture from `AutoConfig` with validation against `SUPPORTED_ARCHITECTURES`
  - Generic extraction works for all 3 archs (same attr names: model.embed_tokens, model.layers, model.norm)
- [x] Add `ModelConfig.from_pretrained()` in `config.py`:
  - Auto-detects `total_layers`, `hidden_dim`, `num_heads`, `num_kv_heads`, `head_dim`, `vocab_size`, `max_seq_len`, `architecture`
  - `SUPPORTED_ARCHITECTURES` set: LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM
  - Fixed `SGLangConfig.max_num_reqs` → `max_running_requests`
- [x] Fixed `total_mem` → `total_memory` in `HealthCheck` (PyTorch API)
- [x] Fixed `build-backend` in `pyproject.toml`: `setuptools.backends._legacy:_Backend` → `setuptools.build_meta`

### Phase 4: Privacy Fixes — Correct DP Guarantees (COMPLETED)
- [x] **CRITICAL BUG FIXED**: Prefill per-token DP accounting in `privacy_engine.py:add_dp_noise()`:
  - Now counts `num_tokens = hidden_states.shape[-2]` and calls `accountant.step()` N times
  - Verified: prefill of 100 tokens → 100 DP steps (was 1), epsilon correctly reflects composition
- [x] Fix sensitivity estimation in `estimate_activation_sensitivity()`:
  - Added `mode` parameter: `"clip_norm"` (default, returns clip_norm directly) vs `"empirical"`
  - Empirical mode uses 99.9th percentile with 1.5x safety margin (was 95th, no margin)
  - Fixed `model.forward_to_layer()` → `model.forward_to_split()`
- [x] Key file permissions in `crypto/channel.py`: `os.chmod(key_path, 0o600)` for ca.key, server.key, client.key
- [x] StreamDecode deduplication in `activation_server.py`: `seen_decode_steps: Set[int]` per session, duplicates logged and skipped

### Phase 5: Production Hardening (COMPLETED)
- [x] Session management in `activation_server.py`: `last_activity` tracking via `session.touch()`, `MAX_SESSIONS=64` enforcement, background cleanup thread evicts sessions idle >10 min
- [x] Graceful shutdown: SIGTERM/SIGINT → `server.stop(grace=5)`, stops cleanup thread, logs final session count
- [x] Client resilience in `client.py`: `_retry_rpc()` with exponential backoff (3 attempts, 1s/2s/4s) for UNAVAILABLE/DEADLINE_EXCEEDED; `close()` method; context manager (`with SplitInferenceClient(...) as client`); `grpc.channel_ready_future()` connectivity check
- [x] CLI configuration in `client.py:main()`: argparse with `--model`, `--local-layers`, `--server-address`, `--dp-epsilon`, `--tls-enabled`, `--max-tokens`; env var overrides `SPLIT_INFERENCE_*`
- [x] Config validation in `config.py`: `SplitInferenceConfig.validate()` checks all parameter ranges (layers, dims, epsilon, delta, ports, etc.)
- [ ] Deployment files: `Dockerfile.local`, `Dockerfile.server`, `docker-compose.yml` (deferred — not needed for research phase)
- [x] Logging: replaced `print()` → `logger.info()` in `launch.py`; structured format in client `main()`

### Phase 6: Testing (COMPLETED — 72 passed, 4 skipped, 0 failed)
- [x] `tests/test_serializer.py` — 8 tests: round-trip shapes `[1,4096]`, `[128,4096]`, `[1,1,4096]`, `[2,64,4096]`, `[1,1]`; float16/float32/bfloat16 dtypes; large tensor; single element
- [x] `tests/test_privacy_engine.py` — 20 tests: clipping (large/small/batch/3D/disabled), noise (gaussian/laplace/disabled/calibration), **N-token accounting** (prefill/decode/batch/cumulative), RDP composition, perturbation reversibility, full pipeline, sensitivity estimation
- [x] `tests/test_grpc_integration.py` — 11 tests: CreateSession (single/multiple), Prefill (success/invalid session), Decode (success/invalid), StreamDecode (5 steps/dedup), HealthCheck (empty/with sessions), full lifecycle
- [x] `tests/test_headless_model.py` — 8 tests: ARCH_REGISTRY (Llama/Mistral/Qwen2/unknown fallback), backwards compat alias, weight loading filter logic, layer count
- [x] `tests/test_multi_arch.py` — 13 tests: SUPPORTED_ARCHITECTURES, ModelConfig defaults/custom, from_pretrained (Qwen2/Mistral, skipped without network), config validation (9 invalid cases), prepare_headless_config

## Verified End-to-End Results (2026-03-18)

### Cross-Machine Inference — WORKING
```
LOCAL:  RTX 4070 Super (Windows) — layers 0-1 + embed + lm_head (3.11 GB)
REMOTE: RTX A6000 (140.113.202.36) — layers 2-27 (6.06B params, ~12 GB)
MODEL:  Qwen/Qwen2-7B-Instruct
```

**Prompt**: "What is the capital of France? Answer in one sentence."
**Output**: "Paris is the capital of France. Paris is the capital city of France."
**Performance**: 12.8 tok/s (cached), 1099ms prefill, 19 tokens

### Quick Start
```bash
# Remote server (asus@140.113.202.36):
cd ~/Split-LLM-Research && source .venv/bin/activate
bash scripts/start_server.sh Qwen/Qwen2-7B-Instruct 2   # GPU 2

# Local machine (Windows):
cd D:\Split-LLM-Research\Split-LLM-Research-main
split-local --model Qwen/Qwen2-7B-Instruct --server-address 140.113.202.36:50051
```

### Performance Benchmarks
| Config | Prefill | Decode | Tok/s |
|--------|---------|--------|-------|
| Same machine (A6000) | 365ms | — | 15.6 |
| Cross-machine (4070↔A6000, cached) | 1099ms | — | 12.8 |
| Cross-machine (4070↔A6000, first run) | 1559ms | — | 4.6 |

### Privacy Verification (Automated Tests)
- Prefill 100 tokens → accountant shows 100 steps (not 1) ✓
- Perturbation round-trip error < 0.001 ✓
- RDP composition: more steps → higher epsilon ✓

## Conventions
- All imports use absolute paths: `from split_inference.config import ...`
- Proto files regenerated via `scripts/build_proto.py` (fixes relative imports automatically)
- SGLang imports wrapped in `try/except ImportError` with `SGLANG_AVAILABLE` flag
- Config uses dataclasses in `config.py`, not separate config files
- HF transformers `dtype=` (not deprecated `torch_dtype=`) for model loading
- `DynamicCache` from `transformers.cache_utils` for KV cache (not legacy tuple format)
- Layer outputs may squeeze batch dim — always restore with `unsqueeze(0)` after layer forward
- `rotary_emb` lives on `base_model.rotary_emb` in newer HF (not on `layer.self_attn`)

## Key Research References
- arxiv:2602.16760: RTT dominates (64%), speculative decoding → 1.2-1.3 tok/step, 15-19 tok/s at 20ms RTT
- SGLang `GenerateReqInput` has `input_embeds` field (GitHub #745, DeepWiki docs)
- SGLang supports `return_hidden_states` via `output["meta_info"]["hidden_states"]`
- Custom model registration: `ModelRegistry.models[name] = class` (process isolation caveat: #11578 — use `auto_map` + `trust_remote_code` instead)
- Balle & Wang 2018: Analytic Gaussian Mechanism for DP noise calibration

## Runtime Notes
- `RemoteModelShard` (direct PyTorch) is the production path — simpler and more reliable than SGLang Engine API
- SGLang Engine integration (`_run_via_engine_api`) is written but untested with real models — kept as optional upgrade path
- Qwen2-7B-Instruct is the default test model (ungated). Llama 3.1 requires HF access approval.
- HF token: stored on both local and remote machines via `huggingface-cli login`
- `accelerate` package required on both sides for `device_map="cpu"` loading

## Known Bugs (All Fixed)
- ~~**Critical**: `privacy_engine.py:add_dp_noise()` counts prefill of N tokens as 1 DP step~~ — **FIXED**: now counts N steps
- ~~`estimate_activation_sensitivity()` calls `model.forward_to_layer()`~~ — **FIXED**: uses `model.forward_to_split()`
- ~~Sensitivity estimation uses 95th percentile~~ — **FIXED**: 99.9th percentile with 1.5x safety margin
- ~~`torch_dtype` deprecated in HF transformers~~ — **FIXED**: uses `dtype=`
- ~~`past_key_value` deprecated in decoder layers~~ — **FIXED**: uses `DynamicCache` + `past_key_values=`
- ~~`rotary_emb` not found on layer attention~~ — **FIXED**: checks `base_model.rotary_emb` first
- ~~Layer output squeezes batch dim~~ — **FIXED**: `unsqueeze(0)` after each layer forward
- ~~`total_mem` attribute error~~ — **FIXED**: `total_memory`
- ~~`setuptools.backends._legacy`~~ — **FIXED**: `setuptools.build_meta`

---
name: Split Inference Architecture
description: Core architecture decisions for privacy-preserving split LLM inference system
type: project
---

Split inference system: local server holds embedding + first K layers + LM head + privacy engine; main server holds layers K..L via RemoteModelShard (direct PyTorch). Communication via gRPC with protobuf serialization.

Two remote execution paths:
1. **RemoteModelShard** (production, verified): Direct HuggingFace layer-by-layer forward with DynamicCache KV caching
2. **SGLang Engine** (optional upgrade): input_embeds + return_hidden_states via Engine.generate() — written but untested

Hardware verified:
- Local: RTX 4070 Super (12GB, Windows) — 3.1GB for Qwen2-7B local shard
- Remote: 3x RTX A6000 (49GB each, Ubuntu) — 12GB for Qwen2-7B remote shard

Multi-architecture: Llama, Mistral, Qwen2 all share same layer accessor paths (model.layers, model.embed_tokens, model.norm). Architecture auto-detected via ModelConfig.from_pretrained().

Key HF compatibility notes:
- rotary_emb lives on base_model (not layer.self_attn) in newer transformers
- Use DynamicCache (not tuple KV format)
- Use dtype= (not torch_dtype=)
- Layer outputs may squeeze batch dim — restore with unsqueeze(0)

**Why:** Research into privacy-preserving LLM inference where raw data never leaves the user's machine.

**How to apply:** All code changes should maintain the invariant that tokens/text/logits stay local. The main server only processes noisy intermediate activations.

---
name: Split Inference Architecture
description: Core architecture decisions for privacy-preserving split LLM inference system
type: project
---

Split inference system: local server holds embedding + first K layers + LM head + privacy engine; main server holds layers K..L via SGLang headless model. Communication via gRPC with mTLS, DP-noised activations only.

Hardware: 3x RTX A6000 (144GB VRAM). Llama 8B at TP=1, 70B at TP=3.

SGLang integration strategy: Engine-level `input_embeds` (Approach A) for proper continuous batching. `GenerateReqInput` has `input_embeds` field. Custom model registered via `ModelRegistry.models[name] = class`.

Multi-architecture scope: Llama, Mistral, Qwen2 (all share same layer accessor paths: model.layers, model.embed_tokens, model.norm, lm_head).

**Why:** Research into privacy-preserving LLM inference where raw data never leaves the user's machine.

**How to apply:** All code changes should maintain the invariant that tokens/text/logits stay local. The main server only processes noisy intermediate activations.

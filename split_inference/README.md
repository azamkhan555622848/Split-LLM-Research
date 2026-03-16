# Privacy-Preserving Split LLM Inference with SGLang

## Architecture Overview

A split-inference system where:
- **Local Server**: Embedding layer + first K transformer blocks + LM head (holds all user data)
- **Main Server**: Remaining transformer blocks served via modified SGLang runtime
- **Communication**: Encrypted intermediate activations over mTLS + gRPC

## Project Structure

```
split_inference/
├── proto/
│   └── split_inference.proto       # gRPC protocol definition
├── local_server/
│   ├── local_model.py              # Local model shard (embedding + first K layers + LM head)
│   ├── privacy_engine.py           # DP noise injection + activation perturbation
│   └── client.py                   # gRPC client that talks to main server
├── main_server/
│   ├── headless_llama.py           # Custom SGLang model (no embedding, starts from hidden states)
│   ├── activation_server.py        # gRPC server receiving encrypted activations
│   └── launch.py                   # SGLang launch script with custom model registration
├── crypto/
│   └── channel.py                  # mTLS setup, encryption utilities
├── config.py                       # Shared configuration
└── README.md
```

## Quick Start

1. Generate TLS certificates (see crypto/channel.py)
2. Start main server: `python main_server/launch.py`
3. Start local server: `python local_server/client.py`

## Key Design Decisions

- Split point K is configurable (default: 2 layers). Research shows 2+ layers makes activation inversion hard (~59% recovery at layer 2, ~35% at layer 8).
- SGLang's ModelRegistry is used to register a custom "headless" model that accepts hidden states instead of token IDs.
- DP noise (Gaussian mechanism) is injected before transmission with configurable epsilon.
- KV cache lives on the main server; only new-token activations are sent per decode step.

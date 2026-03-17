"""
Launch Script for Main Server.

Sets up:
1. Headless model config with auto_map (process-isolation-safe registration)
2. Modified config.json for the headless architecture
3. gRPC activation server (parallel to SGLang's HTTP server)
4. mTLS certificate validation

Usage:
    split-server --model meta-llama/Llama-3.1-8B-Instruct \
                 --local-layers 2 \
                 --tp 1 \
                 --port 50051 \
                 --tls-enabled
"""
import os
import json
import shutil
import logging
import argparse
import threading
from pathlib import Path

from split_inference.config import SplitInferenceConfig, ModelConfig, SUPPORTED_ARCHITECTURES

logger = logging.getLogger(__name__)

# The headless model module that will be copied into the model directory
# for auto_map / trust_remote_code loading (survives process isolation)
HEADLESS_MODEL_MODULE = "split_inference.main_server.headless_llama"


def prepare_headless_config(
    original_model_path: str,
    local_layers: int,
    output_dir: str = "/tmp/headless_model",
) -> str:
    """
    Create a modified model config for the headless architecture.

    Changes to config.json:
    1. architectures: ["HeadlessTransformerForRemoteInference"]
    2. auto_map: points to the headless model module (process-isolation-safe)
    3. Custom fields: local_layers, is_headless, original_architecture
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load original config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(original_model_path)
    config_dict = config.to_dict()

    # Detect original architecture
    original_arch = "LlamaForCausalLM"
    if "architectures" in config_dict and config_dict["architectures"]:
        original_arch = config_dict["architectures"][0]
    if original_arch not in SUPPORTED_ARCHITECTURES:
        logger.warning(f"Architecture {original_arch} not in supported set, proceeding anyway")

    # Modify for headless operation
    config_dict["architectures"] = ["HeadlessTransformerForRemoteInference"]
    config_dict["original_architecture"] = original_arch
    config_dict["local_layers"] = local_layers
    config_dict["is_headless"] = True

    # auto_map for process-isolation-safe model loading
    # SGLang workers will import from the model directory via trust_remote_code
    config_dict["auto_map"] = {
        "AutoModelForCausalLM": "headless_model.HeadlessTransformerForRemoteInference",
    }

    # Save modified config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy the headless model module into the output directory
    # so trust_remote_code can import it
    _write_headless_model_module(output_dir, original_arch, local_layers)

    # Symlink weights from original path (avoid copying large files)
    original_path = Path(original_model_path)
    if original_path.exists():
        for pattern in ("*.safetensors", "*.bin", "tokenizer*", "*.model"):
            for src_file in original_path.glob(pattern):
                link_path = Path(output_dir) / src_file.name
                if not link_path.exists():
                    os.symlink(src_file, link_path)

    logger.info(f"Headless config prepared at: {output_dir}")
    logger.info(f"  Original architecture: {original_arch}")
    logger.info(f"  Local layers: {local_layers} (skipped on remote)")
    logger.info(f"  Remote layers: {config_dict['num_hidden_layers'] - local_layers}")

    return output_dir


def _write_headless_model_module(output_dir: str, original_arch: str, local_layers: int):
    """
    Write a headless_model.py into the model directory for trust_remote_code loading.
    This re-exports HeadlessTransformerForRemoteInference from the installed package.
    """
    module_content = '''"""
Auto-generated headless model module for trust_remote_code loading.
Re-exports HeadlessTransformerForRemoteInference from the split_inference package.
"""
from split_inference.main_server.headless_llama import (
    HeadlessTransformerForRemoteInference,
    HeadlessTransformerModel,
    EntryClass,
)

__all__ = [
    "HeadlessTransformerForRemoteInference",
    "HeadlessTransformerModel",
    "EntryClass",
]
'''
    module_path = os.path.join(output_dir, "headless_model.py")
    with open(module_path, "w") as f:
        f.write(module_content)
    logger.info(f"Wrote headless_model.py to {module_path}")


def launch_grpc_server(config: SplitInferenceConfig):
    """Launch the gRPC activation server in a background thread."""
    from split_inference.main_server.activation_server import serve

    thread = threading.Thread(
        target=serve,
        args=(config,),
        daemon=True,
        name="grpc-activation-server",
    )
    thread.start()
    logger.info("gRPC activation server started in background thread")
    return thread


def launch_sglang_server(model_path: str, config: SplitInferenceConfig):
    """
    Launch SGLang's HTTP server with the headless model.

    In production, the gRPC server is the primary interface.
    The HTTP server is secondary (debugging/monitoring).
    """
    try:
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(
            model_path=model_path,
            tp_size=config.sglang.tp_size,
            dp_size=config.sglang.dp_size,
            port=config.sglang.sglang_port,
            mem_fraction_static=config.sglang.mem_fraction,
            max_running_requests=config.sglang.max_running_requests,
            chunked_prefill_size=config.sglang.chunked_prefill_size,
            disable_radix_cache=not config.sglang.enable_radix_cache,
            quantization=config.sglang.quantization,
            trust_remote_code=True,
        )
    except ImportError:
        logger.warning("SGLang not available - running gRPC server only")


def main():
    parser = argparse.ArgumentParser(
        description="Launch Privacy-Preserving Split Inference Main Server"
    )
    parser.add_argument(
        "--model", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model path or ID",
    )
    parser.add_argument(
        "--local-layers", type=int, default=2,
        help="Number of layers on the local server (split point K)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (1 for 8B, 3 for 70B)",
    )
    parser.add_argument(
        "--grpc-port", type=int, default=50051,
        help="gRPC server port for activation exchange",
    )
    parser.add_argument(
        "--sglang-port", type=int, default=30000,
        help="SGLang HTTP server port (for monitoring)",
    )
    parser.add_argument(
        "--tls-enabled", action="store_true",
        help="Enable mTLS for gRPC",
    )
    parser.add_argument(
        "--quantization", type=str, default=None,
        choices=["fp8", "awq", "gptq", None],
        help="Quantization method for remote layers",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Auto-detect model config
    model_config = ModelConfig.from_pretrained(
        args.model,
        local_layers=args.local_layers,
    )

    # Build config
    config = SplitInferenceConfig()
    config.model = model_config
    config.sglang.tp_size = args.tp
    config.sglang.sglang_port = args.sglang_port
    config.sglang.quantization = args.quantization
    config.network.main_server_port = args.grpc_port
    config.network.tls_enabled = args.tls_enabled

    config.validate()

    logger.info("=" * 60)
    logger.info("Privacy-Preserving Split Inference - Main Server")
    logger.info("=" * 60)
    logger.info(f"  Model:         {args.model}")
    logger.info(f"  Architecture:  {model_config.architecture}")
    logger.info(f"  Split point:   Layer {args.local_layers}")
    logger.info(f"  Remote layers: {config.remote_layers}")
    logger.info(f"  TP:            {args.tp}")
    logger.info(f"  gRPC port:     {args.grpc_port} ({'mTLS' if args.tls_enabled else 'insecure'})")
    logger.info(f"  SGLang port:   {args.sglang_port}")
    logger.info(f"  Quantization:  {args.quantization or 'none'}")
    logger.info("=" * 60)

    # 1. Prepare headless model config (with auto_map for process isolation)
    headless_path = prepare_headless_config(
        args.model,
        args.local_layers,
    )

    # 2. Start gRPC activation server
    grpc_thread = launch_grpc_server(config)

    # 3. Start SGLang server (blocking)
    launch_sglang_server(headless_path, config)


if __name__ == "__main__":
    main()

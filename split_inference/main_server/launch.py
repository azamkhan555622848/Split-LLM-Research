"""
Launch Script for Main Server.

Sets up:
1. Custom HeadlessLlama model registration in SGLang's ModelRegistry
2. Modified config.json for the headless architecture
3. gRPC activation server (parallel to SGLang's HTTP server)
4. mTLS certificate validation

Usage:
    python launch.py --model meta-llama/Llama-3.1-8B-Instruct \
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

from split_inference.config import SplitInferenceConfig

logger = logging.getLogger(__name__)


def prepare_headless_config(
    original_model_path: str,
    local_layers: int,
    output_dir: str = "/tmp/headless_model",
) -> str:
    """
    Create a modified model config for the headless architecture.
    
    Changes to config.json:
    1. architectures: ["HeadlessLlamaForRemoteInference"]
    2. Add custom fields: local_layers, is_headless
    
    The actual weights are loaded from the original path,
    with HeadlessLlamaForRemoteInference.load_weights() filtering
    out embedding/lm_head/local-layer weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy original config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(original_model_path)
    config_dict = config.to_dict()
    
    # Modify architecture
    config_dict["architectures"] = ["HeadlessLlamaForRemoteInference"]
    config_dict["local_layers"] = local_layers
    config_dict["is_headless"] = True
    
    # Save modified config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Symlink weights from original path (avoid copying large files)
    # SGLang will load from this directory, and our load_weights()
    # will filter out weights that don't belong on the remote server
    original_path = Path(original_model_path)
    if original_path.exists():
        for weight_file in original_path.glob("*.safetensors"):
            link_path = Path(output_dir) / weight_file.name
            if not link_path.exists():
                os.symlink(weight_file, link_path)
        
        # Also link tokenizer files (SGLang needs them even for headless)
        for tok_file in original_path.glob("tokenizer*"):
            link_path = Path(output_dir) / tok_file.name
            if not link_path.exists():
                os.symlink(tok_file, link_path)
    
    logger.info(f"Headless config prepared at: {output_dir}")
    logger.info(f"  Architecture: HeadlessLlamaForRemoteInference")
    logger.info(f"  Local layers: {local_layers} (skipped on remote)")
    logger.info(f"  Remote layers: {config_dict['num_hidden_layers'] - local_layers}")
    
    return output_dir


def register_headless_model():
    """Register HeadlessLlamaForRemoteInference with SGLang's ModelRegistry."""
    try:
        from sglang.srt.models.registry import ModelRegistry
        from main_server.headless_llama import HeadlessLlamaForRemoteInference
        
        ModelRegistry.models["HeadlessLlamaForRemoteInference"] = \
            HeadlessLlamaForRemoteInference
        
        logger.info("✅ HeadlessLlamaForRemoteInference registered in ModelRegistry")
        return True
    except ImportError as e:
        logger.error(f"SGLang not available: {e}")
        return False


def launch_grpc_server(config: SplitInferenceConfig):
    """Launch the gRPC activation server in a background thread."""
    from main_server.activation_server import serve
    
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
    
    This provides the standard OpenAI-compatible API on top of our
    headless model, which can be useful for debugging and monitoring.
    
    In production, the gRPC server is the primary interface for
    split inference. The HTTP server is secondary.
    """
    try:
        from sglang.srt.entrypoints.http_server import launch_server
        
        launch_server(
            model_path=model_path,
            tp_size=config.sglang.tp_size,
            dp_size=config.sglang.dp_size,
            port=config.sglang.sglang_port,
            mem_fraction_static=config.sglang.mem_fraction,
            max_num_reqs=config.sglang.max_num_reqs,
            chunked_prefill_size=config.sglang.chunked_prefill_size,
            disable_radix_cache=not config.sglang.enable_radix_cache,
            quantization=config.sglang.quantization,
        )
    except ImportError:
        logger.warning("SGLang not available — running gRPC server only")


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
        help="Tensor parallelism degree",
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
    
    # Build config
    config = SplitInferenceConfig()
    config.model.model_name = args.model
    config.model.local_layers = args.local_layers
    config.sglang.tp_size = args.tp
    config.sglang.sglang_port = args.sglang_port
    config.sglang.quantization = args.quantization
    config.network.main_server_port = args.grpc_port
    config.network.tls_enabled = args.tls_enabled
    
    print("=" * 60)
    print("🔒 Privacy-Preserving Split Inference — Main Server")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Split point:   Layer {args.local_layers}")
    print(f"  Remote layers: {config.remote_layers}")
    print(f"  TP:            {args.tp}")
    print(f"  gRPC port:     {args.grpc_port} ({'mTLS' if args.tls_enabled else 'insecure'})")
    print(f"  SGLang port:   {args.sglang_port}")
    print(f"  Quantization:  {args.quantization or 'none'}")
    print("=" * 60)
    
    # 1. Register custom model
    register_headless_model()
    
    # 2. Prepare headless model config
    headless_path = prepare_headless_config(
        args.model,
        args.local_layers,
    )
    
    # 3. Start gRPC activation server
    grpc_thread = launch_grpc_server(config)
    
    # 4. Start SGLang server (blocking)
    launch_sglang_server(headless_path, config)


if __name__ == "__main__":
    main()

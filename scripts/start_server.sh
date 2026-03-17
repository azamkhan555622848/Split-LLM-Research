#!/bin/bash
# Start the split inference main server on the remote GPU machine.
# Usage: bash scripts/start_server.sh [--model MODEL] [--gpu GPU_ID]

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${1:-Qwen/Qwen2-7B-Instruct}"
GPU="${2:-0}"
LOCAL_LAYERS=2
GRPC_PORT=50051

export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================================"
echo "Split Inference — Main Server"
echo "============================================================"
echo "  Model:      $MODEL"
echo "  GPU:        $GPU"
echo "  Layers:     $LOCAL_LAYERS local (skipped), rest remote"
echo "  gRPC port:  $GRPC_PORT"
echo "============================================================"

python -c "
import logging, signal, grpc, torch, time, threading
from concurrent import futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

from split_inference.config import SplitInferenceConfig, ModelConfig
from split_inference.main_server.activation_server import SplitInferenceServicer
from split_inference.proto import split_inference_pb2_grpc as pb2_grpc

MODEL = '$MODEL'
LOCAL_LAYERS = $LOCAL_LAYERS
PORT = $GRPC_PORT

mc = ModelConfig.from_pretrained(MODEL, local_layers=LOCAL_LAYERS)
cfg = SplitInferenceConfig()
cfg.model = mc
cfg.network.tls_enabled = False
cfg.network.main_server_host = '0.0.0.0'
cfg.network.main_server_port = PORT

server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    options=[
        ('grpc.max_send_message_length', 64*1024*1024),
        ('grpc.max_receive_message_length', 64*1024*1024),
    ],
)
servicer = SplitInferenceServicer(cfg)
pb2_grpc.add_SplitInferenceServiceServicer_to_server(servicer, server)
server.add_insecure_port(f'0.0.0.0:{PORT}')

shutdown = threading.Event()
def handler(sig, frame):
    logging.info('Shutting down...')
    server.stop(5)
    servicer.processor._cleanup_stop.set()
    shutdown.set()
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

server.start()
logging.info(f'Server ready on 0.0.0.0:{PORT}')
shutdown.wait()
"

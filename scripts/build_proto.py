#!/usr/bin/env python3
"""Compile protobuf definitions for split inference gRPC service."""
import subprocess
import sys
from pathlib import Path


def build():
    project_root = Path(__file__).resolve().parent.parent
    proto_dir = project_root / "split_inference" / "proto"
    proto_file = proto_dir / "split_inference.proto"

    if not proto_file.exists():
        print(f"ERROR: Proto file not found: {proto_file}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
        f"--pyi_out={proto_dir}",
        str(proto_file),
    ]

    print(f"Compiling: {proto_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: protoc failed:\n{result.stderr}")
        sys.exit(1)

    # Fix the import in the generated grpc file to use relative imports
    grpc_file = proto_dir / "split_inference_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        content = content.replace(
            "import split_inference_pb2 as split__inference__pb2",
            "from . import split_inference_pb2 as split__inference__pb2",
        )
        grpc_file.write_text(content)

    print("Generated:")
    for f in proto_dir.glob("split_inference_pb2*"):
        print(f"  {f.name}")
    print("Done.")


if __name__ == "__main__":
    build()

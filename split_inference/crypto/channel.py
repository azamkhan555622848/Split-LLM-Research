"""
Cryptographic Utilities for Split Inference.

Provides:
1. mTLS certificate generation (CA, server, client certs)
2. Activation encryption helpers (for transport layer)
3. TEE attestation stubs (for production deployment)

In the split inference architecture:
- Transport encryption: mTLS on the gRPC channel (always on)
- Application-level: DP noise (privacy_engine.py)
- Hardware-level: TEE (optional, Intel SGX / AMD SEV / NVIDIA CC)

For research prototypes, mTLS + DP noise is sufficient.
For production with untrusted cloud: add TEE attestation.
"""
import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_certificates(
    output_dir: str = "certs",
    ca_cn: str = "SplitInference CA",
    server_cn: str = "split-inference-server",
    client_cn: str = "split-inference-client",
    days: int = 365,
):
    """
    Generate a full mTLS certificate chain using OpenSSL.
    
    Creates:
        certs/
        ├── ca.pem          # CA certificate
        ├── ca.key          # CA private key
        ├── server.pem      # Server certificate (signed by CA)
        ├── server.key      # Server private key
        ├── client.pem      # Client certificate (signed by CA)
        └── client.key      # Client private key
    
    In production:
    - Use a proper PKI infrastructure
    - Store keys in HSM/Vault
    - Rotate certificates regularly
    - Use ECDSA P-256 or Ed25519 instead of RSA
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ca_key = os.path.join(output_dir, "ca.key")
    ca_cert = os.path.join(output_dir, "ca.pem")
    server_key = os.path.join(output_dir, "server.key")
    server_csr = os.path.join(output_dir, "server.csr")
    server_cert = os.path.join(output_dir, "server.pem")
    client_key = os.path.join(output_dir, "client.key")
    client_csr = os.path.join(output_dir, "client.csr")
    client_cert = os.path.join(output_dir, "client.pem")
    
    # 1. Generate CA key and self-signed certificate
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "ec",
        "-pkeyopt", "ec_paramgen_curve:P-256",
        "-days", str(days),
        "-nodes",
        "-keyout", ca_key,
        "-out", ca_cert,
        "-subj", f"/CN={ca_cn}",
    ], check=True, capture_output=True)
    logger.info(f"Generated CA: {ca_cert}")
    
    # 2. Generate server key and CSR
    subprocess.run([
        "openssl", "req", "-newkey", "ec",
        "-pkeyopt", "ec_paramgen_curve:P-256",
        "-nodes",
        "-keyout", server_key,
        "-out", server_csr,
        "-subj", f"/CN={server_cn}",
    ], check=True, capture_output=True)
    
    # Create server extensions file for SAN
    server_ext = os.path.join(output_dir, "server.ext")
    with open(server_ext, "w") as f:
        f.write(
            "authorityKeyIdentifier=keyid,issuer\n"
            "basicConstraints=CA:FALSE\n"
            "subjectAltName=@alt_names\n"
            "\n"
            "[alt_names]\n"
            "DNS.1=localhost\n"
            "DNS.2=split-inference-server\n"
            "IP.1=127.0.0.1\n"
            "IP.2=0.0.0.0\n"
        )
    
    # Sign server cert with CA
    subprocess.run([
        "openssl", "x509", "-req",
        "-in", server_csr,
        "-CA", ca_cert,
        "-CAkey", ca_key,
        "-CAcreateserial",
        "-out", server_cert,
        "-days", str(days),
        "-extfile", server_ext,
    ], check=True, capture_output=True)
    logger.info(f"Generated server cert: {server_cert}")
    
    # 3. Generate client key and CSR
    subprocess.run([
        "openssl", "req", "-newkey", "ec",
        "-pkeyopt", "ec_paramgen_curve:P-256",
        "-nodes",
        "-keyout", client_key,
        "-out", client_csr,
        "-subj", f"/CN={client_cn}",
    ], check=True, capture_output=True)
    
    # Sign client cert with CA
    subprocess.run([
        "openssl", "x509", "-req",
        "-in", client_csr,
        "-CA", ca_cert,
        "-CAkey", ca_key,
        "-CAcreateserial",
        "-out", client_cert,
        "-days", str(days),
    ], check=True, capture_output=True)
    logger.info(f"Generated client cert: {client_cert}")
    
    # Cleanup CSR and extension files
    for f in [server_csr, client_csr, server_ext,
              os.path.join(output_dir, "ca.srl")]:
        if os.path.exists(f):
            os.remove(f)
    
    print(f"\n✅ mTLS certificates generated in {output_dir}/")
    print(f"   CA:     {ca_cert}")
    print(f"   Server: {server_cert} + {server_key}")
    print(f"   Client: {client_cert} + {client_key}")
    
    return {
        "ca_cert": ca_cert,
        "server_cert": server_cert,
        "server_key": server_key,
        "client_cert": client_cert,
        "client_key": client_key,
    }


# ============================================================================
# TEE Attestation Stubs
# ============================================================================

class TEEAttestationError(Exception):
    pass


def verify_tee_attestation(attestation_report: bytes) -> bool:
    """
    Verify that the remote server is running inside a Trusted Execution Environment.
    
    In production, this would:
    1. Verify the attestation report signature (from Intel/AMD/NVIDIA)
    2. Check the enclave measurement matches the expected binary
    3. Validate the TEE's identity and freshness (anti-replay)
    
    Supported TEEs for LLM inference:
    - Intel TDX (Trust Domain Extensions) — for CPU-based inference
    - AMD SEV-SNP — for VM-level isolation
    - NVIDIA Confidential Computing — for GPU inference (H100+)
    
    References:
    - SecureInfer (arXiv:2510.19979) — TEE-GPU architecture
    - Chrapek et al., "Performance and cost across CPU and GPU TEEs" (2025)
    """
    # Stub — in production, use:
    # - Intel DCAP for TDX attestation
    # - AMD SEV-SNP attestation via sev-tool
    # - NVIDIA CC attestation via nvidia-attestation-sdk
    logger.warning(
        "TEE attestation not implemented — "
        "using transport encryption only. "
        "For production, integrate TEE attestation SDK."
    )
    return True


# ============================================================================
# Main: Generate certs for development
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_certificates()

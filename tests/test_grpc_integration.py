"""Tests for gRPC integration — full lifecycle and error handling."""
import pytest
import torch
import grpc
from concurrent import futures
from unittest.mock import patch

from split_inference.config import SplitInferenceConfig
from split_inference.local_server.client import ActivationSerializer
from split_inference.proto import split_inference_pb2 as pb2
from split_inference.proto import split_inference_pb2_grpc as pb2_grpc
import split_inference.main_server.activation_server as srv_mod


@pytest.fixture
def grpc_server():
    """Start a gRPC server with SGLang bypassed for testing."""
    with patch.object(srv_mod, 'SGLANG_AVAILABLE', False):
        from split_inference.main_server.activation_server import SplitInferenceServicer

        config = SplitInferenceConfig()
        config.network.tls_enabled = False

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=[
                ('grpc.max_send_message_length', 64 * 1024 * 1024),
                ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            ],
        )
        servicer = SplitInferenceServicer(config)
        pb2_grpc.add_SplitInferenceServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port('localhost:0')  # Random available port
        server.start()

        channel = grpc.insecure_channel(
            f'localhost:{port}',
            options=[
                ('grpc.max_send_message_length', 64 * 1024 * 1024),
                ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            ],
        )
        stub = pb2_grpc.SplitInferenceServiceStub(channel)

        yield stub, servicer

        servicer.processor._cleanup_stop.set()
        server.stop(0)


@pytest.fixture
def ser():
    return ActivationSerializer()


class TestCreateSession:
    def test_create_session_success(self, grpc_server):
        stub, _ = grpc_server
        resp = stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test-model', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(dp_enabled=True, epsilon=8.0),
        ))
        assert resp.success
        assert len(resp.session_id) > 0

    def test_create_multiple_sessions(self, grpc_server):
        stub, servicer = grpc_server
        ids = set()
        for _ in range(5):
            resp = stub.CreateSession(pb2.CreateSessionRequest(
                model_name='test', local_layers=2, max_seq_len=4096,
                dp_metadata=pb2.DPMetadata(),
            ))
            assert resp.success
            ids.add(resp.session_id)
        assert len(ids) == 5
        assert len(servicer.processor.sessions) == 5


class TestPrefill:
    def test_prefill_success(self, grpc_server, ser):
        stub, _ = grpc_server
        resp = stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(),
        ))
        sid = resp.session_id

        h = torch.randn(128, 4096)
        resp = stub.Prefill(pb2.PrefillRequest(
            session_id=sid,
            hidden_states=ser.serialize(h, 'float16'),
            position_ids=list(range(128)),
            seq_len=128, hidden_dim=4096, dtype='float16', noise_sigma=0.1,
        ))
        assert resp.success
        out = ser.deserialize(resp.hidden_states, 'float16', 'cpu')
        assert out.shape == (128, 4096)
        assert resp.total_time_ms > 0

    def test_prefill_invalid_session(self, grpc_server, ser):
        stub, _ = grpc_server
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.Prefill(pb2.PrefillRequest(
                session_id='nonexistent',
                hidden_states=b'x',
            ))
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestDecode:
    def test_decode_success(self, grpc_server, ser):
        stub, _ = grpc_server
        resp = stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(),
        ))
        sid = resp.session_id

        h = torch.randn(1, 4096)
        resp = stub.Decode(pb2.DecodeRequest(
            session_id=sid,
            hidden_states=ser.serialize(h, 'float16'),
            position_id=128, hidden_dim=4096, dtype='float16',
            decode_step=1, noise_sigma=0.05,
        ))
        assert resp.success
        out = ser.deserialize(resp.hidden_states, 'float16', 'cpu')
        assert out.shape == (1, 4096)

    def test_decode_invalid_session(self, grpc_server, ser):
        stub, _ = grpc_server
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.Decode(pb2.DecodeRequest(
                session_id='bad-session',
                hidden_states=ser.serialize(torch.randn(1, 4096), 'float16'),
                position_id=0, hidden_dim=4096, dtype='float16',
                decode_step=0, noise_sigma=0.0,
            ))
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


class TestStreamDecode:
    def test_stream_decode_multiple_steps(self, grpc_server, ser):
        stub, _ = grpc_server
        resp = stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(),
        ))
        sid = resp.session_id

        def requests():
            for step in range(5):
                yield pb2.DecodeRequest(
                    session_id=sid,
                    hidden_states=ser.serialize(torch.randn(1, 4096), 'float16'),
                    position_id=100 + step, hidden_dim=4096, dtype='float16',
                    decode_step=step, noise_sigma=0.05,
                )

        responses = list(stub.StreamDecode(requests()))
        assert len(responses) == 5
        assert all(r.success for r in responses)

    def test_stream_decode_deduplication(self, grpc_server, ser):
        stub, _ = grpc_server
        resp = stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(),
        ))
        sid = resp.session_id

        def requests():
            for step in [0, 1, 1, 2, 3]:  # step 1 duplicated
                yield pb2.DecodeRequest(
                    session_id=sid,
                    hidden_states=ser.serialize(torch.randn(1, 4096), 'float16'),
                    position_id=100 + step, hidden_dim=4096, dtype='float16',
                    decode_step=step, noise_sigma=0.05,
                )

        responses = list(stub.StreamDecode(requests()))
        assert len(responses) == 4  # Duplicate step 1 skipped


class TestHealthCheck:
    def test_health_check(self, grpc_server):
        stub, _ = grpc_server
        resp = stub.HealthCheck(pb2.HealthCheckRequest())
        assert resp.healthy
        assert resp.active_sessions == 0

    def test_health_check_with_sessions(self, grpc_server):
        stub, _ = grpc_server
        stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(),
        ))
        resp = stub.HealthCheck(pb2.HealthCheckRequest())
        assert resp.healthy
        assert resp.active_sessions == 1


class TestFullLifecycle:
    def test_create_prefill_decode_health(self, grpc_server, ser):
        """Full lifecycle: CreateSession -> Prefill -> Decode x3 -> HealthCheck."""
        stub, _ = grpc_server

        # Create
        resp = stub.CreateSession(pb2.CreateSessionRequest(
            model_name='test', local_layers=2, max_seq_len=4096,
            dp_metadata=pb2.DPMetadata(dp_enabled=True, epsilon=8.0),
        ))
        assert resp.success
        sid = resp.session_id

        # Prefill
        h = torch.randn(32, 4096)
        resp = stub.Prefill(pb2.PrefillRequest(
            session_id=sid,
            hidden_states=ser.serialize(h, 'float16'),
            position_ids=list(range(32)),
            seq_len=32, hidden_dim=4096, dtype='float16', noise_sigma=0.1,
        ))
        assert resp.success

        # Decode x3
        for step in range(3):
            h1 = torch.randn(1, 4096)
            resp = stub.Decode(pb2.DecodeRequest(
                session_id=sid,
                hidden_states=ser.serialize(h1, 'float16'),
                position_id=32 + step, hidden_dim=4096, dtype='float16',
                decode_step=step + 1, noise_sigma=0.05,
            ))
            assert resp.success

        # HealthCheck
        resp = stub.HealthCheck(pb2.HealthCheckRequest())
        assert resp.healthy
        assert resp.active_sessions == 1

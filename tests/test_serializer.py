"""Tests for ActivationSerializer — round-trip shapes and dtype handling."""
import pytest
import torch
import numpy as np

from split_inference.local_server.client import ActivationSerializer


class TestActivationSerializer:
    """Round-trip serialization tests for various shapes and dtypes."""

    @pytest.fixture
    def ser(self):
        return ActivationSerializer()

    @pytest.mark.parametrize("shape", [
        (1, 4096),
        (128, 4096),
        (1, 1, 4096),
        (2, 64, 4096),
        (1, 1),
    ])
    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_round_trip_shapes(self, ser, shape, dtype):
        """Tensor survives serialize -> deserialize with correct shape."""
        original = torch.randn(*shape)
        data = ser.serialize(original, dtype=dtype)
        restored = ser.deserialize(data, dtype=dtype, device="cpu")
        assert restored.shape == original.shape

    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_round_trip_values_float(self, ser, dtype):
        """Values are preserved within dtype precision."""
        original = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        data = ser.serialize(original, dtype=dtype)
        restored = ser.deserialize(data, dtype=dtype, device="cpu")

        if dtype == "float16":
            assert torch.allclose(original.half().float(), restored.float(), atol=1e-3)
        else:
            assert torch.allclose(original, restored, atol=1e-6)

    def test_bfloat16_serializes_as_fp16(self, ser):
        """bfloat16 is converted to fp16 on the wire."""
        original = torch.randn(4, 128)
        data = ser.serialize(original, dtype="bfloat16")
        restored = ser.deserialize(data, dtype="bfloat16", device="cpu")
        assert restored.shape == original.shape

    def test_unknown_dtype_defaults_to_fp16(self, ser):
        """Unknown dtype string falls back to float16."""
        original = torch.randn(2, 64)
        data = ser.serialize(original, dtype="unknown_type")
        restored = ser.deserialize(data, dtype="unknown_type", device="cpu")
        assert restored.shape == original.shape

    def test_large_tensor(self, ser):
        """Serialization works for large activation tensors."""
        original = torch.randn(512, 4096)
        data = ser.serialize(original, dtype="float16")
        restored = ser.deserialize(data, dtype="float16", device="cpu")
        assert restored.shape == (512, 4096)
        # fp16 wire: 1 (ndim) + 8 (shape) + 512*4096*2 (data)
        assert len(data) == 1 + 8 + 512 * 4096 * 2

    def test_single_element(self, ser):
        """Edge case: 1-element tensor."""
        original = torch.tensor([42.0])
        data = ser.serialize(original, dtype="float16")
        restored = ser.deserialize(data, dtype="float16", device="cpu")
        assert restored.shape == (1,)

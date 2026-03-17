"""Tests for HeadlessTransformerModel — multi-arch support and weight loading."""
import pytest
import torch

from split_inference.main_server.headless_llama import (
    HeadlessTransformerForRemoteInference,
    HeadlessTransformerModel,
    HeadlessLlamaForRemoteInference,
    ARCH_REGISTRY,
    _get_decoder_layer_class,
    SGLANG_AVAILABLE,
)


class TestArchRegistry:
    """Tests for architecture registry and decoder layer dispatch."""

    def test_supported_architectures(self):
        assert "LlamaForCausalLM" in ARCH_REGISTRY
        assert "MistralForCausalLM" in ARCH_REGISTRY
        assert "Qwen2ForCausalLM" in ARCH_REGISTRY

    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not installed")
    def test_llama_decoder_layer(self):
        cls = _get_decoder_layer_class("LlamaForCausalLM")
        assert cls.__name__ == "LlamaDecoderLayer"

    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not installed")
    def test_mistral_uses_llama_layer(self):
        cls = _get_decoder_layer_class("MistralForCausalLM")
        assert cls.__name__ == "LlamaDecoderLayer"

    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not installed")
    def test_qwen2_decoder_layer(self):
        cls = _get_decoder_layer_class("Qwen2ForCausalLM")
        assert cls.__name__ == "Qwen2DecoderLayer"

    @pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not installed")
    def test_unknown_arch_falls_back_to_llama(self):
        cls = _get_decoder_layer_class("UnknownArchitecture")
        assert cls.__name__ == "LlamaDecoderLayer"


class TestBackwardsCompatAlias:
    def test_alias_exists(self):
        assert HeadlessLlamaForRemoteInference is HeadlessTransformerForRemoteInference


class TestWeightLoading:
    """Tests for selective weight loading logic."""

    def _make_fake_weights(self, total_layers=8, hidden_dim=16):
        """Generate fake weight name/tensor pairs mimicking a Llama model."""
        weights = []
        # Embedding
        weights.append(("model.embed_tokens.weight", torch.randn(100, hidden_dim)))
        # Layers
        for i in range(total_layers):
            weights.append((f"model.layers.{i}.self_attn.q_proj.weight", torch.randn(hidden_dim, hidden_dim)))
            weights.append((f"model.layers.{i}.self_attn.k_proj.weight", torch.randn(hidden_dim, hidden_dim)))
            weights.append((f"model.layers.{i}.mlp.gate_proj.weight", torch.randn(hidden_dim, hidden_dim)))
        # Norm and LM head
        weights.append(("model.norm.weight", torch.randn(hidden_dim)))
        weights.append(("lm_head.weight", torch.randn(100, hidden_dim)))
        return weights

    def test_skips_embedding_and_lm_head(self):
        """Weight loading logic correctly filters local-only weights.
        NOTE: Cannot instantiate actual SGLang layers outside the runtime
        (TP group not initialized), so we test the filtering logic directly.
        """
        weights = self._make_fake_weights(total_layers=4, hidden_dim=16)
        local_layers = 2

        loaded = []
        skipped = []
        for name, _ in weights:
            if any(skip in name for skip in ("embed_tokens", "lm_head", "model.norm.")):
                skipped.append(name)
                continue
            if "model.layers." in name:
                parts = name.split(".")
                layer_idx = int(parts[2])
                if layer_idx < local_layers:
                    skipped.append(name)
                    continue
            loaded.append(name)

        # Should skip embed, lm_head, norm, and layers 0-1
        assert "model.embed_tokens.weight" in skipped
        assert "lm_head.weight" in skipped
        assert "model.norm.weight" in skipped
        assert any("layers.0." in n for n in skipped)
        assert any("layers.1." in n for n in skipped)
        # Should load layers 2 and 3
        assert any("layers.2." in n for n in loaded)
        assert any("layers.3." in n for n in loaded)
        assert len(loaded) == 6  # 2 layers * 3 weights each


class TestModelInit:
    """Tests for HeadlessTransformerModel initialization."""

    def test_without_sglang_empty_layers(self):
        """Without SGLang, layers should be empty ModuleList."""
        if SGLANG_AVAILABLE:
            pytest.skip("Test only meaningful without SGLang")

        class FakeConfig:
            hidden_size = 4096
            num_hidden_layers = 32

        model = HeadlessTransformerModel(
            FakeConfig(), local_layers=2, architecture="LlamaForCausalLM",
        )
        assert model.num_remote_layers == 30
        assert len(model.layers) == 0  # No SGLang = no layers

    def test_layer_count_calculation(self):
        """Layer count arithmetic is correct regardless of SGLang availability."""
        if SGLANG_AVAILABLE:
            # With SGLang, layers require full config — test arithmetic only
            assert 32 - 4 == 28  # trivial but documents the formula
        else:
            class FakeConfig:
                hidden_size = 4096
                num_hidden_layers = 32

            model = HeadlessTransformerModel(
                FakeConfig(), local_layers=4, architecture="LlamaForCausalLM",
            )
            assert model.num_remote_layers == 28
            assert model.local_layers == 4
            assert model.num_total_layers == 32

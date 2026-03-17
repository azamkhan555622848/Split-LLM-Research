"""Tests for multi-architecture support — ModelConfig, launch, local_model."""
import json
import os
import tempfile
import pytest

from split_inference.config import (
    ModelConfig,
    SplitInferenceConfig,
    SUPPORTED_ARCHITECTURES,
)


class TestSupportedArchitectures:
    def test_includes_llama(self):
        assert "LlamaForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_includes_mistral(self):
        assert "MistralForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_includes_qwen2(self):
        assert "Qwen2ForCausalLM" in SUPPORTED_ARCHITECTURES


class TestModelConfig:
    def test_default_values(self):
        mc = ModelConfig()
        assert mc.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert mc.total_layers == 32
        assert mc.local_layers == 2
        assert mc.hidden_dim == 4096

    def test_custom_values(self):
        mc = ModelConfig(
            model_name="custom/model",
            architecture="Qwen2ForCausalLM",
            total_layers=28,
            local_layers=4,
            hidden_dim=3584,
            num_kv_heads=4,
        )
        assert mc.architecture == "Qwen2ForCausalLM"
        assert mc.total_layers == 28
        assert mc.num_kv_heads == 4


class TestModelConfigFromPretrained:
    """Tests for auto-detection from HuggingFace configs."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_HF_TESTS"),
        reason="Requires network access to HuggingFace (set RUN_HF_TESTS=1)",
    )
    def test_qwen2(self):
        mc = ModelConfig.from_pretrained("Qwen/Qwen2-7B-Instruct", local_layers=2)
        assert mc.architecture == "Qwen2ForCausalLM"
        assert mc.total_layers == 28
        assert mc.hidden_dim == 3584
        assert mc.num_kv_heads == 4

    @pytest.mark.skipif(
        not os.environ.get("RUN_HF_TESTS"),
        reason="Requires network access to HuggingFace (set RUN_HF_TESTS=1)",
    )
    def test_mistral(self):
        mc = ModelConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", local_layers=2)
        assert mc.architecture == "MistralForCausalLM"
        assert mc.total_layers == 32
        assert mc.hidden_dim == 4096
        assert mc.num_kv_heads == 8


class TestConfigValidation:
    def test_valid_config(self):
        config = SplitInferenceConfig()
        config.validate()  # Should not raise

    def test_invalid_local_layers_zero(self):
        config = SplitInferenceConfig()
        config.model.local_layers = 0
        with pytest.raises(ValueError, match="local_layers"):
            config.validate()

    def test_invalid_local_layers_exceeds_total(self):
        config = SplitInferenceConfig()
        config.model.local_layers = 100
        with pytest.raises(ValueError, match="local_layers"):
            config.validate()

    def test_invalid_epsilon(self):
        config = SplitInferenceConfig()
        config.privacy.dp_epsilon = -1
        with pytest.raises(ValueError, match="dp_epsilon"):
            config.validate()

    def test_invalid_delta(self):
        config = SplitInferenceConfig()
        config.privacy.dp_delta = 2.0
        with pytest.raises(ValueError, match="dp_delta"):
            config.validate()

    def test_invalid_mechanism(self):
        config = SplitInferenceConfig()
        config.privacy.dp_mechanism = "unknown"
        with pytest.raises(ValueError, match="dp_mechanism"):
            config.validate()

    def test_invalid_port(self):
        config = SplitInferenceConfig()
        config.network.main_server_port = 0
        with pytest.raises(ValueError, match="main_server_port"):
            config.validate()

    def test_invalid_tp_size(self):
        config = SplitInferenceConfig()
        config.sglang.tp_size = 0
        with pytest.raises(ValueError, match="tp_size"):
            config.validate()

    def test_invalid_mem_fraction(self):
        config = SplitInferenceConfig()
        config.sglang.mem_fraction = 1.5
        with pytest.raises(ValueError, match="mem_fraction"):
            config.validate()


class TestPrepareHeadlessConfig:
    """Tests for launch.py:prepare_headless_config()."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_HF_TESTS"),
        reason="Requires network access to HuggingFace (set RUN_HF_TESTS=1)",
    )
    def test_generates_correct_config(self):
        from split_inference.main_server.launch import prepare_headless_config

        with tempfile.TemporaryDirectory() as tmpdir:
            path = prepare_headless_config(
                "Qwen/Qwen2-7B-Instruct", local_layers=2, output_dir=tmpdir,
            )

            with open(os.path.join(path, "config.json")) as f:
                cfg = json.load(f)

            assert cfg["architectures"] == ["HeadlessTransformerForRemoteInference"]
            assert cfg["original_architecture"] == "Qwen2ForCausalLM"
            assert cfg["local_layers"] == 2
            assert cfg["is_headless"] is True
            assert "auto_map" in cfg
            assert os.path.exists(os.path.join(path, "headless_model.py"))

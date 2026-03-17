"""Tests for PrivacyEngine — clipping, noise, accounting, perturbation."""
import math
import pytest
import torch
import numpy as np

from split_inference.config import PrivacyConfig
from split_inference.local_server.privacy_engine import (
    PrivacyEngine,
    PrivacyAccountant,
    estimate_activation_sensitivity,
)


class TestActivationClipping:
    """Tests for clip_activations()."""

    def test_clips_large_vectors(self):
        config = PrivacyConfig(clip_enabled=True, clip_norm=1.0)
        engine = PrivacyEngine(config, hidden_dim=4)
        # Vector with norm >> 1
        h = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        clipped = engine.clip_activations(h)
        norm = torch.norm(clipped, p=2, dim=-1)
        assert norm.item() <= 1.0 + 1e-6

    def test_preserves_small_vectors(self):
        config = PrivacyConfig(clip_enabled=True, clip_norm=10.0)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        clipped = engine.clip_activations(h)
        assert torch.allclose(h, clipped, atol=1e-6)

    def test_batch_clipping(self):
        config = PrivacyConfig(clip_enabled=True, clip_norm=5.0)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.randn(32, 4) * 100  # All vectors way over norm
        clipped = engine.clip_activations(h)
        norms = torch.norm(clipped, p=2, dim=-1)
        assert (norms <= 5.0 + 1e-5).all()

    def test_disabled_clipping(self):
        config = PrivacyConfig(clip_enabled=False)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        result = engine.clip_activations(h)
        assert torch.equal(h, result)

    def test_3d_input(self):
        config = PrivacyConfig(clip_enabled=True, clip_norm=2.0)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.randn(2, 16, 4) * 50
        clipped = engine.clip_activations(h)
        flat = clipped.reshape(-1, 4)
        norms = torch.norm(flat, p=2, dim=-1)
        assert (norms <= 2.0 + 1e-5).all()


class TestDPNoise:
    """Tests for add_dp_noise() and noise calibration."""

    def test_gaussian_noise_added(self):
        config = PrivacyConfig(dp_enabled=True, dp_mechanism="gaussian", dp_epsilon=8.0)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.zeros(1, 4)
        noisy, sigma = engine.add_dp_noise(h)
        assert sigma > 0
        assert not torch.equal(noisy, h)

    def test_laplace_noise_added(self):
        config = PrivacyConfig(dp_enabled=True, dp_mechanism="laplace", dp_epsilon=8.0)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.zeros(1, 4)
        noisy, sigma = engine.add_dp_noise(h)
        assert sigma > 0
        assert not torch.equal(noisy, h)

    def test_dp_disabled_no_noise(self):
        config = PrivacyConfig(dp_enabled=False)
        engine = PrivacyEngine(config, hidden_dim=4)
        h = torch.ones(1, 4)
        noisy, sigma = engine.add_dp_noise(h)
        assert sigma == 0.0
        assert torch.equal(noisy, h)

    def test_sigma_calibration(self):
        """Lower epsilon -> higher sigma (more noise)."""
        s_high_eps = PrivacyEngine._calibrate_gaussian_sigma(1.0, 10.0, 1e-5)
        s_low_eps = PrivacyEngine._calibrate_gaussian_sigma(1.0, 1.0, 1e-5)
        assert s_low_eps > s_high_eps


class TestPerTokenAccounting:
    """Tests for the CRITICAL bug fix: N-token prefill = N DP steps."""

    def test_prefill_counts_n_steps(self):
        config = PrivacyConfig(dp_enabled=True, dp_epsilon=8.0)
        engine = PrivacyEngine(config, hidden_dim=128)
        h = torch.randn(100, 128)
        engine.add_dp_noise(h)
        assert engine.accountant.total_steps == 100

    def test_decode_counts_1_step(self):
        config = PrivacyConfig(dp_enabled=True, dp_epsilon=8.0)
        engine = PrivacyEngine(config, hidden_dim=128)
        h = torch.randn(1, 128)
        engine.add_dp_noise(h)
        assert engine.accountant.total_steps == 1

    def test_batch_uses_seq_dim(self):
        """shape[-2] is the token count for [B, seq_len, D]."""
        config = PrivacyConfig(dp_enabled=True, dp_epsilon=8.0)
        engine = PrivacyEngine(config, hidden_dim=128)
        h = torch.randn(2, 50, 128)
        engine.add_dp_noise(h)
        assert engine.accountant.total_steps == 50

    def test_cumulative_accounting(self):
        config = PrivacyConfig(dp_enabled=True, dp_epsilon=8.0)
        engine = PrivacyEngine(config, hidden_dim=128)
        engine.add_dp_noise(torch.randn(64, 128))   # prefill
        engine.add_dp_noise(torch.randn(1, 128))    # decode 1
        engine.add_dp_noise(torch.randn(1, 128))    # decode 2
        assert engine.accountant.total_steps == 66


class TestRDPAccountant:
    """Tests for RDP-based privacy accounting."""

    def test_rdp_composition(self):
        acc = PrivacyAccountant(epsilon_per_step=8.0, delta_per_step=1e-5)
        for _ in range(10):
            acc.step(sigma=1.0, sensitivity=1.0)
        eps = acc.get_total_epsilon(target_delta=1e-5)
        assert eps > 0
        assert acc.total_steps == 10

    def test_more_steps_higher_epsilon(self):
        acc1 = PrivacyAccountant()
        acc2 = PrivacyAccountant()
        for _ in range(10):
            acc1.step(sigma=1.0)
        for _ in range(100):
            acc2.step(sigma=1.0)
        assert acc2.get_total_epsilon() > acc1.get_total_epsilon()

    def test_budget_report(self):
        acc = PrivacyAccountant(epsilon_per_step=8.0, delta_per_step=1e-5)
        acc.step(sigma=0.5)
        report = acc.get_budget_report()
        assert "total_steps" in report
        assert "total_epsilon" in report
        assert report["total_steps"] == 1


class TestPerturbation:
    """Tests for reversible structured perturbation."""

    def test_perturbation_reversibility(self):
        config = PrivacyConfig(
            dp_enabled=False,
            perturbation_enabled=True,
            perturbation_seed=42,
            perturbation_scale=0.1,
        )
        engine = PrivacyEngine(config, hidden_dim=128)
        original = torch.randn(8, 128)
        perturbed = engine.add_perturbation(original.clone(), step=3)
        recovered = PrivacyEngine.remove_perturbation(
            perturbed, perturbation_seed=42, perturbation_scale=0.1, step=3,
        )
        assert torch.allclose(original, recovered, atol=1e-3)

    def test_different_steps_different_perturbation(self):
        config = PrivacyConfig(
            dp_enabled=False,
            perturbation_enabled=True,
            perturbation_seed=42,
            perturbation_scale=1.0,
        )
        engine = PrivacyEngine(config, hidden_dim=64)
        h = torch.zeros(4, 64)
        p1 = engine.add_perturbation(h.clone(), step=0)
        p2 = engine.add_perturbation(h.clone(), step=1)
        assert not torch.equal(p1, p2)

    def test_perturbation_disabled(self):
        config = PrivacyConfig(perturbation_enabled=False)
        engine = PrivacyEngine(config, hidden_dim=64)
        h = torch.ones(4, 64)
        result = engine.add_perturbation(h.clone(), step=0)
        assert torch.equal(h, result)


class TestFullProtectPipeline:
    """Tests for the full protect() pipeline."""

    def test_protect_clips_and_adds_noise(self):
        config = PrivacyConfig(
            dp_enabled=True, clip_enabled=True, clip_norm=5.0,
            dp_epsilon=8.0, dp_mechanism="gaussian",
        )
        engine = PrivacyEngine(config, hidden_dim=128)
        h = torch.randn(16, 128) * 100
        protected, sigma = engine.protect(h, step=0)
        assert sigma > 0
        assert protected.shape == h.shape
        assert engine.accountant.total_steps == 16

    def test_privacy_report(self):
        config = PrivacyConfig(dp_enabled=True, dp_epsilon=4.0)
        engine = PrivacyEngine(config, hidden_dim=64)
        engine.protect(torch.randn(10, 64), step=0)
        report = engine.get_privacy_report()
        assert report["mechanism"] == "gaussian"
        assert report["total_steps"] == 10
        assert report["total_epsilon"] > 0
        assert report["sigma"] > 0


class TestSensitivityEstimation:
    """Tests for estimate_activation_sensitivity()."""

    def test_clip_norm_mode(self):
        sens = estimate_activation_sensitivity(None, None, [], 2, mode="clip_norm")
        assert sens == 10.0

    def test_invalid_mode_raises(self):
        with pytest.raises(Exception):
            estimate_activation_sensitivity(None, None, [], 2, mode="nonexistent")

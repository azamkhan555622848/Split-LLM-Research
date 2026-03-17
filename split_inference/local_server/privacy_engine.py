"""
Privacy Engine for Split Inference.

Implements:
1. Activation clipping (bound L2 norm before noise injection)
2. Gaussian mechanism DP noise injection (calibrated to epsilon, delta)
3. Structured perturbation (reversible noise using shared seed)
4. Per-request privacy accounting

References:
- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
- arxiv:2602.16760 - Privacy-Aware Split Inference with Speculative Decoding (2025)
- Fission (IACR ePrint 2025/653)
"""
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple

from split_inference.config import PrivacyConfig


@dataclass
class PrivacyAccountant:
    """
    Track cumulative privacy budget across decode steps.
    Uses the simple composition theorem: after T steps with (ε, δ)-DP each,
    the total is (√(2T ln(1/δ')) · ε + T·ε·(e^ε - 1), T·δ + δ')-DP.
    
    For tighter bounds, use Rényi DP composition (RDP).
    """
    epsilon_per_step: float = 8.0
    delta_per_step: float = 1e-5
    total_steps: int = 0
    
    # RDP-based accounting (tighter bounds)
    rdp_orders: list = field(default_factory=lambda: [
        1.5, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256
    ])
    rdp_epsilons: list = field(default_factory=list)  # Per-order accumulated RDP epsilon
    
    def __post_init__(self):
        if not self.rdp_epsilons:
            self.rdp_epsilons = [0.0] * len(self.rdp_orders)
    
    def step(self, sigma: float, sensitivity: float = 1.0):
        """Account for one Gaussian mechanism application."""
        self.total_steps += 1
        
        # RDP of Gaussian mechanism: ε(α) = α * sensitivity² / (2 * σ²)
        for i, alpha in enumerate(self.rdp_orders):
            rdp_eps = alpha * (sensitivity ** 2) / (2 * sigma ** 2)
            self.rdp_epsilons[i] += rdp_eps
    
    def get_total_epsilon(self, target_delta: float = 1e-5) -> float:
        """
        Convert accumulated RDP to (ε, δ)-DP using the optimal conversion.
        Returns the tightest ε for the given δ.
        """
        min_epsilon = float('inf')
        for alpha, rdp_eps in zip(self.rdp_orders, self.rdp_epsilons):
            # RDP to (ε, δ)-DP conversion
            epsilon = rdp_eps + math.log(1 / target_delta) / (alpha - 1)
            min_epsilon = min(min_epsilon, epsilon)
        return min_epsilon
    
    def get_budget_report(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "total_epsilon": self.get_total_epsilon(),
            "target_delta": 1e-5,
            "per_step_sigma": self.epsilon_per_step,
        }


class PrivacyEngine:
    """
    Applies privacy-preserving transformations to activation tensors
    before they leave the local server.
    
    Pipeline:
        raw_activation
            → clip_norm()          # Bound sensitivity
            → add_dp_noise()       # Gaussian/Laplace mechanism
            → add_perturbation()   # Optional structured noise (reversible)
            → serialize & send
    """
    
    def __init__(self, config: PrivacyConfig, hidden_dim: int):
        self.config = config
        self.hidden_dim = hidden_dim
        self.accountant = PrivacyAccountant(
            epsilon_per_step=config.dp_epsilon,
            delta_per_step=config.dp_delta,
        )
        
        # Compute Gaussian noise sigma from epsilon/delta
        # σ = sensitivity * √(2 ln(1.25/δ)) / ε
        if config.dp_mechanism == "gaussian":
            self.sigma = self._calibrate_gaussian_sigma(
                config.dp_sensitivity,
                config.dp_epsilon,
                config.dp_delta,
            )
        elif config.dp_mechanism == "laplace":
            # b = sensitivity / ε
            self.sigma = config.dp_sensitivity / config.dp_epsilon
        else:
            raise ValueError(f"Unknown mechanism: {config.dp_mechanism}")
        
        # Perturbation generator (seeded for reversibility)
        if config.perturbation_enabled:
            self.perturbation_rng = np.random.RandomState(config.perturbation_seed)
    
    @staticmethod
    def _calibrate_gaussian_sigma(
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """
        Calibrate Gaussian noise σ for (ε, δ)-DP.
        
        Analytic Gaussian Mechanism (Balle & Wang, 2018):
        σ ≥ sensitivity * √(2 ln(1.25/δ)) / ε
        """
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    def clip_activations(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Clip per-token activation vectors to bounded L2 norm.
        This is CRITICAL for DP guarantees — without clipping,
        the sensitivity is unbounded and noise calibration is meaningless.
        
        Args:
            hidden_states: [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        
        Returns:
            Clipped tensor with same shape, each vector has ||v||₂ ≤ clip_norm
        """
        if not self.config.clip_enabled:
            return hidden_states
        
        C = self.config.clip_norm
        
        # Compute per-vector L2 norms
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_dim)  # [N, D]
        norms = torch.norm(flat, p=2, dim=1, keepdim=True)  # [N, 1]
        
        # Scale down vectors that exceed the clip norm
        scale = torch.clamp(C / (norms + 1e-8), max=1.0)  # [N, 1]
        clipped = flat * scale  # [N, D]
        
        return clipped.reshape(original_shape)
    
    def add_dp_noise(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Add calibrated DP noise to activation tensors.

        IMPORTANT: Each token's activation is an independent query to the
        mechanism, so prefill of N tokens must count as N DP steps, not 1.

        Args:
            hidden_states: [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]

        Returns:
            (noisy_tensor, sigma_used)
        """
        if not self.config.dp_enabled:
            return hidden_states, 0.0

        if self.config.dp_mechanism == "gaussian":
            noise = torch.randn_like(hidden_states) * self.sigma
        elif self.config.dp_mechanism == "laplace":
            u = torch.rand_like(hidden_states) - 0.5
            noise = -self.sigma * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
        else:
            raise ValueError(f"Unknown mechanism: {self.config.dp_mechanism}")

        noisy = hidden_states + noise

        # Account for EACH token as a separate DP step.
        # For prefill with N tokens, this correctly records N steps.
        # shape[-2] is seq_len for both [seq_len, D] and [B, seq_len, D].
        num_tokens = hidden_states.shape[-2] if hidden_states.ndim >= 2 else 1
        for _ in range(num_tokens):
            self.accountant.step(self.sigma, self.config.dp_sensitivity)

        return noisy, self.sigma
    
    def add_perturbation(
        self,
        hidden_states: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Add structured (reversible) perturbation.
        
        Uses a seeded PRNG so the server can generate the same noise
        and subtract it. This provides an additional layer of obfuscation
        on top of DP noise, but does NOT provide DP guarantees by itself.
        
        The seed is derived from: base_seed XOR step_counter
        so each decode step gets unique but reproducible perturbation.
        """
        if not self.config.perturbation_enabled:
            return hidden_states
        
        # Derive per-step seed
        step_seed = self.config.perturbation_seed ^ (step * 2654435761)  # Knuth hash
        rng = np.random.RandomState(step_seed % (2**31))
        
        # Generate structured perturbation
        perturbation = torch.from_numpy(
            rng.randn(*hidden_states.shape).astype(np.float16)
        ).to(hidden_states.device) * self.config.perturbation_scale
        
        return hidden_states + perturbation
    
    @staticmethod
    def remove_perturbation(
        hidden_states: torch.Tensor,
        perturbation_seed: int,
        perturbation_scale: float,
        step: int,
    ) -> torch.Tensor:
        """
        Remove structured perturbation on the server side.
        Called by the server after receiving activations (if perturbation is enabled).
        """
        step_seed = perturbation_seed ^ (step * 2654435761)
        rng = np.random.RandomState(step_seed % (2**31))
        
        perturbation = torch.from_numpy(
            rng.randn(*hidden_states.shape).astype(np.float16)
        ).to(hidden_states.device) * perturbation_scale
        
        return hidden_states - perturbation
    
    def protect(
        self,
        hidden_states: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, float]:
        """
        Full privacy pipeline: clip → DP noise → perturbation.
        
        Args:
            hidden_states: Raw activation tensor from local model
            step: Decode step counter (for perturbation seed derivation)
        
        Returns:
            (protected_tensor, sigma_used)
        """
        # Step 1: Clip activations to bound sensitivity
        h = self.clip_activations(hidden_states)
        
        # Step 2: Add DP noise (Gaussian or Laplace)
        h, sigma = self.add_dp_noise(h)
        
        # Step 3: Add reversible perturbation (optional)
        h = self.add_perturbation(h, step)
        
        return h, sigma
    
    def get_privacy_report(self) -> dict:
        """Return current privacy budget consumption."""
        report = self.accountant.get_budget_report()
        report.update({
            "mechanism": self.config.dp_mechanism,
            "sigma": self.sigma,
            "clip_norm": self.config.clip_norm if self.config.clip_enabled else "disabled",
            "perturbation": "enabled" if self.config.perturbation_enabled else "disabled",
        })
        return report


# ============================================================================
# Utility: Measure empirical sensitivity of activations
# ============================================================================

def estimate_activation_sensitivity(
    model,
    tokenizer,
    sample_texts: list,
    split_layer: int,
    num_pairs: int = 100,
    mode: str = "clip_norm",
) -> float:
    """
    Estimate L2 sensitivity of activations at the split layer.

    Two modes:
    - "clip_norm" (default, conservative): Returns the clip norm directly.
      If clipping is applied before noise, the sensitivity is exactly clip_norm.
      This is the correct approach when clip_activations() is in the pipeline.
    - "empirical": Empirically measure sensitivity by perturbing one token
      and measuring activation L2 distance. Uses 99.9th percentile with
      1.5x safety margin for robustness.

    Args:
        model: The local model shard (LocalModelShard)
        tokenizer: Tokenizer
        sample_texts: Representative input texts
        split_layer: Layer index at which to measure
        num_pairs: Number of input pairs to test
        mode: "clip_norm" or "empirical"

    Returns:
        Estimated L2 sensitivity
    """
    if mode == "clip_norm":
        # When clipping is enabled, sensitivity = clip_norm by definition
        clip_norm = getattr(model, 'config', None)
        if clip_norm and hasattr(clip_norm, 'clip_norm'):
            return clip_norm.clip_norm
        return 10.0  # Default clip norm

    # Empirical mode
    sensitivities = []

    for text in sample_texts[:num_pairs]:
        tokens = tokenizer.encode(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            h_original = model.forward_to_split(tokens)

        # Perturbed activation (change one random token)
        perturbed = tokens.clone()
        pos = np.random.randint(0, tokens.shape[1])
        perturbed[0, pos] = np.random.randint(0, tokenizer.vocab_size)

        with torch.no_grad():
            h_perturbed = model.forward_to_split(perturbed)

        # L2 distance
        diff = torch.norm(h_original - h_perturbed, p=2).item()
        sensitivities.append(diff)

    # Use 99.9th percentile with 1.5x safety margin
    return float(np.percentile(sensitivities, 99.9)) * 1.5

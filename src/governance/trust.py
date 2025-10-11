# src/governance/trust.py
"""
Implements dynamic trust scoring for committed patches using a Beta-Bernoulli model.
Corresponds to Section IX-B of the paper.
"""
from typing import Dict, Any


class TrustScorer:
    """
    Maintains and updates a trust score ρ for each committed edit.
    """

    def __init__(self, config: Dict[str, Any]):
        self.alpha = config.get('alpha', 2)  # Prior successes
        self.beta = config.get('beta', 1)  # Prior failures
        self.patch_reputations: Dict[str, Dict] = {}  # Stores s and f for each patch

    def initialize_trust(self, patch_id: str) -> float:
        """
        Initializes a patch with s=0, f=0 and computes its initial trust score.
        """
        self.patch_reputations[patch_id] = {'s': 0, 'f': 0}
        initial_rho = self.alpha / (self.alpha + self.beta)
        print(f"TRUST: Initialized trust for patch {patch_id}. Initial ρ = {initial_rho:.2f}")
        return initial_rho

    def update_trust_score(self, patch_id: str, success: bool) -> float:
        """
        Updates the success (s) or failure (f) count and recomputes the trust score.
        Based on the Bayesian update model from Eq. 20.
        """
        if patch_id not in self.patch_reputations:
            self.initialize_trust(patch_id)

        if success:
            self.patch_reputations[patch_id]['s'] += 1
        else:
            self.patch_reputations[patch_id]['f'] += 1

        s = self.patch_reputations[patch_id]['s']
        f = self.patch_reputations[patch_id]['f']

        # Beta-Bernoulli update formula
        rho = (s + self.alpha) / (s + f + self.alpha + self.beta)

        print(
            f"TRUST: Updated patch {patch_id} ({'success' if success else 'failure'}). New ρ = {rho:.2f} (s={s}, f={f})")
        return rho
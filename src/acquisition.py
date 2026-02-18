"""
Acquisition functions for intervention selection: Expected Information Gain (EIG).
"""

import numpy as np
from typing import Optional
from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief


def entropy(belief: np.ndarray) -> float:
    b = belief[belief > 0]
    return -np.sum(b * np.log(b))


def expected_information_gain(scm: LinearGaussianSCM, belief: GraphBelief,
                               intervention_value: float = 2.0,
                               n_simulations: int = 500,
                               n_samples_per_sim: int = 50,
                               sigma2: float = 0.3,
                               seed: Optional[int] = None) -> dict:
    """Compute EIG for each possible intervention target."""
    rng = np.random.default_rng(seed)
    current_ent = entropy(belief.belief)
    eig = {}

    for target in scm.variables:
        post_ent_sum = 0.0
        for _ in range(n_simulations):
            s = rng.integers(0, 2**31)
            sim_data = scm.sample_interventional(target, intervention_value, n_samples_per_sim, seed=s)

            log_liks = np.zeros(belief.K)
            for k in range(belief.K):
                w = belief.estimate_weights(k, sim_data, target)
                log_liks[k] = belief.compute_log_likelihood(k, sim_data, target, w, sigma2)

            log_p = log_liks + np.log(belief.belief + 1e-300)
            log_p -= log_p.max()
            p = np.exp(log_p)
            p /= p.sum()
            post_ent_sum += entropy(p)

        eig[target] = current_ent - post_ent_sum / n_simulations

    return eig


def select_intervention(scm: LinearGaussianSCM, belief: GraphBelief,
                        intervention_value: float = 2.0,
                        n_simulations: int = 500,
                        n_samples_per_sim: int = 50,
                        sigma2: float = 0.3,
                        seed: Optional[int] = None) -> tuple:
    """Select intervention target maximising EIG. Returns (target, scores)."""
    scores = expected_information_gain(scm, belief, intervention_value, n_simulations,
                                       n_samples_per_sim, sigma2, seed)
    best = max(scores, key=scores.get)
    return best, scores


def random_intervention(scm: LinearGaussianSCM, seed: Optional[int] = None) -> str:
    rng = np.random.default_rng(seed)
    return rng.choice(scm.variables)


if __name__ == "__main__":
    scm = LinearGaussianSCM()
    belief = GraphBelief(tau=3.0)
    print(f"Current entropy: {entropy(belief.belief):.4f}")
    best, scores = select_intervention(scm, belief, n_simulations=100, seed=42)
    print("EIG scores:")
    for v, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  do({v}): {s:.4f}")
    print(f"Best: do({best})")

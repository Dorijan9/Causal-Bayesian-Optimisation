"""
Graph belief module: prior from LLM, likelihood computation, Bayesian posterior update.
"""

import json
import numpy as np
from scipy.stats import norm
from typing import Optional


class GraphBelief:
    """Probability distribution over candidate causal graphs."""

    def __init__(self, candidates_path: str = "data/candidate_graphs.json", tau: float = 3.0):
        with open(candidates_path) as f:
            data = json.load(f)

        self.variable_order = data["variable_order"]
        self.n_vars = len(self.variable_order)
        self.var_to_idx = {v: i for i, v in enumerate(self.variable_order)}

        self.candidates = []
        self.confidences = []
        for c in data["candidates"]:
            self.candidates.append({
                "id": c["id"],
                "confidence": c["confidence"],
                "adjacency": np.array(c["adjacency_matrix"]),
                "edges": c["edges"],
                "rationale": c["rationale"],
            })
            self.confidences.append(c["confidence"])

        self.K = len(self.candidates)
        self.confidences = np.array(self.confidences)
        self.tau = tau
        self.prior = self._softmax_prior(tau)
        self.belief = self.prior.copy()
        self.belief_history = [self.belief.copy()]

    def _softmax_prior(self, tau: float) -> np.ndarray:
        if tau == 0:
            return np.ones(self.K) / self.K
        logits = tau * self.confidences
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def get_parents(self, graph_idx: int, var: str) -> list:
        j = self.var_to_idx[var]
        adj = self.candidates[graph_idx]["adjacency"]
        return [v for i, v in enumerate(self.variable_order) if adj[i, j] != 0]

    def estimate_weights(self, graph_idx: int, data: np.ndarray, target_var: str) -> np.ndarray:
        """Estimate edge weights via OLS for a candidate graph."""
        weights = np.zeros((self.n_vars, self.n_vars))
        for var in self.variable_order:
            if var == target_var:
                continue
            j = self.var_to_idx[var]
            parents = self.get_parents(graph_idx, var)
            if not parents:
                continue
            pidx = [self.var_to_idx[p] for p in parents]
            X = data[:, pidx]
            y = data[:, j]
            try:
                w = np.linalg.lstsq(X, y, rcond=None)[0]
                for k, pi in enumerate(pidx):
                    weights[pi, j] = w[k]
            except np.linalg.LinAlgError:
                pass
        return weights

    def compute_log_likelihood(self, graph_idx: int, data: np.ndarray,
                                target_var: str, weights: np.ndarray = None,
                                sigma2: float = 0.3) -> float:
        """Log-likelihood of interventional data under a candidate graph."""
        if weights is None:
            weights = self.estimate_weights(graph_idx, data, target_var)

        ll = 0.0
        for var in self.variable_order:
            if var == target_var:
                continue
            j = self.var_to_idx[var]
            parents = self.get_parents(graph_idx, var)
            if not parents:
                ll += norm.logpdf(data[:, j], 0, np.sqrt(1.0)).sum()
            else:
                mean = sum(weights[self.var_to_idx[p], j] * data[:, self.var_to_idx[p]] for p in parents)
                ll += norm.logpdf(data[:, j], mean, np.sqrt(sigma2)).sum()
        return ll

    def update(self, intv_data: np.ndarray, target_var: str,
               sigma2: float = 0.3, all_data: np.ndarray = None) -> np.ndarray:
        """Bayesian update: posterior ∝ likelihood × prior."""
        est_data = all_data if all_data is not None else intv_data
        log_liks = np.zeros(self.K)
        for k in range(self.K):
            w = self.estimate_weights(k, est_data, target_var)
            log_liks[k] = self.compute_log_likelihood(k, intv_data, target_var, w, sigma2)

        log_post = log_liks + np.log(self.belief + 1e-300)
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        self.belief = post
        self.belief_history.append(post.copy())
        return post

    def entropy(self) -> float:
        b = self.belief[self.belief > 0]
        return -np.sum(b * np.log(b))

    def map_estimate(self) -> int:
        return int(np.argmax(self.belief))

    def has_converged(self, threshold: float = 0.95) -> bool:
        return self.belief.max() > threshold

    def summary(self) -> dict:
        mi = self.map_estimate()
        return {
            "belief": self.belief.tolist(),
            "map_graph": self.candidates[mi]["id"],
            "map_probability": float(self.belief[mi]),
            "entropy": float(self.entropy()),
            "converged": self.has_converged(),
        }


if __name__ == "__main__":
    gb = GraphBelief(tau=3.0)
    print("Prior:", gb.prior.round(4))
    for k, c in enumerate(gb.candidates):
        print(f"  {c['id']}: conf={c['confidence']}, prior={gb.prior[k]:.4f}, parents_of_C={gb.get_parents(k, 'C')}")

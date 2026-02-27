"""
Graph belief module: Bayesian inference over candidate causal graphs.

Key design: zero-mean Gaussian priors on edge weights, with posterior
weight estimation via Bayesian linear regression and marginal likelihood
computation by integrating out the weights analytically.

The LLM contributes structural knowledge (which edges exist) via the
graph prior; parametric knowledge (sign, magnitude of weights) comes
entirely from interventional data.
"""

import json
import numpy as np
from scipy.stats import norm, multivariate_normal
from typing import Optional


class WeightPosterior:
    """Bayesian posterior over edge weights for a single child variable.

    Prior:      w_j ~ N(0, sigma_w^2 * I)
    Likelihood: x_j | X_pa ~ N(w_j^T X_pa, sigma_eps^2)
    Posterior:  w_j | data ~ N(m_j, Sigma_j)
    """

    def __init__(self, n_parents: int, sigma_w2: float, sigma_eps2: float):
        self.n_parents = n_parents
        self.sigma_w2 = sigma_w2
        self.sigma_eps2 = sigma_eps2

        # Prior: N(0, sigma_w^2 * I)
        self.prior_precision = (1.0 / sigma_w2) * np.eye(n_parents)
        self.prior_mean = np.zeros(n_parents)

        # Initialise posterior = prior
        self.precision = self.prior_precision.copy()
        self.mean = self.prior_mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.precision)

    def update(self, X_parents: np.ndarray, y_child: np.ndarray):
        """Update posterior given data.

        Posterior precision:  Lambda = (1/sigma_w^2) I + (1/sigma_eps^2) X^T X
        Posterior mean:       m = Lambda^{-1} (1/sigma_eps^2) X^T y
        """
        self.precision = self.prior_precision + (1.0 / self.sigma_eps2) * X_parents.T @ X_parents
        self.mean = np.linalg.solve(
            self.precision,
            (1.0 / self.sigma_eps2) * X_parents.T @ y_child
        )

    def log_marginal_likelihood(self, X_parents: np.ndarray, y_child: np.ndarray) -> float:
        """Log marginal likelihood with weights integrated out.

        P(y | X, G_k) = N(y | 0, sigma_eps^2 I + sigma_w^2 X X^T)
        """
        n = len(y_child)
        # Predictive covariance: sigma_eps^2 I + sigma_w^2 X X^T
        cov = self.sigma_eps2 * np.eye(n) + self.sigma_w2 * (X_parents @ X_parents.T)
        cov += 1e-8 * np.eye(n)  # jitter for stability

        try:
            return float(multivariate_normal.logpdf(y_child, mean=np.zeros(n), cov=cov))
        except np.linalg.LinAlgError:
            cov += 1e-4 * np.eye(n)
            return float(multivariate_normal.logpdf(y_child, mean=np.zeros(n), cov=cov))

    def sample_weights(self, rng: np.random.Generator) -> np.ndarray:
        """Draw weights from current posterior."""
        return rng.multivariate_normal(self.mean, self.covariance)


class GraphBelief:
    """Bayesian belief over candidate causal graphs with weight posteriors.

    Two-level inference:
    1. Graph level: P(G_k | data) via marginal likelihood with weights integrated out
    2. Weight level: P(w | G_k, data) via Bayesian linear regression (zero-mean prior)
    """

    def __init__(self, candidates_path: str = "data/candidate_graphs.json",
                 tau: float = 3.0, sigma_w2: float = 0.5, sigma_eps2: float = 0.3):
        with open(candidates_path) as f:
            data = json.load(f)

        self.variable_order = data["variable_order"]
        self.n_vars = len(self.variable_order)
        self.var_to_idx = {v: i for i, v in enumerate(self.variable_order)}
        self.sigma_w2 = sigma_w2
        self.sigma_eps2 = sigma_eps2

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

        # Initialise weight posteriors for each (graph, variable) pair
        self._init_weight_posteriors()

    def _softmax_prior(self, tau: float) -> np.ndarray:
        if tau == 0:
            return np.ones(self.K) / self.K
        logits = tau * self.confidences
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def _init_weight_posteriors(self):
        """Create a WeightPosterior for each (graph, child variable) pair."""
        self.weight_posteriors = []
        for k in range(self.K):
            graph_wp = {}
            for var in self.variable_order:
                parents = self.get_parents(k, var)
                n_pa = len(parents)
                if n_pa > 0:
                    graph_wp[var] = WeightPosterior(n_pa, self.sigma_w2, self.sigma_eps2)
            self.weight_posteriors.append(graph_wp)

    def get_parents(self, graph_idx: int, var: str) -> list:
        j = self.var_to_idx[var]
        adj = self.candidates[graph_idx]["adjacency"]
        return [v for i, v in enumerate(self.variable_order) if adj[i, j] != 0]

    def _get_parent_data(self, graph_idx: int, var: str,
                         data: np.ndarray) -> Optional[np.ndarray]:
        """Extract parent columns from data for a given variable under graph_idx."""
        parents = self.get_parents(graph_idx, var)
        if not parents:
            return None
        pidx = [self.var_to_idx[p] for p in parents]
        return data[:, pidx]

    def update_weight_posteriors(self, graph_idx: int, data: np.ndarray,
                                 target_var: str):
        """Update weight posteriors for graph_idx given accumulated data."""
        for var in self.variable_order:
            if var == target_var:
                continue
            X_pa = self._get_parent_data(graph_idx, var, data)
            if X_pa is None:
                continue
            j = self.var_to_idx[var]
            y = data[:, j]
            self.weight_posteriors[graph_idx][var].update(X_pa, y)

    def compute_log_marginal_likelihood(self, graph_idx: int, data: np.ndarray,
                                         target_var: str) -> float:
        """Log marginal likelihood P(D | G_k) integrating out weights.

        Decomposes across variables:
        log P(D | G_k) = sum_{j != target} log P(y_j | X_pa(j), G_k)
        """
        ll = 0.0
        for var in self.variable_order:
            if var == target_var:
                continue
            j = self.var_to_idx[var]
            y = data[:, j]
            X_pa = self._get_parent_data(graph_idx, var, data)

            if X_pa is None:
                # Root variable: P(x_j) = N(0, 1)
                ll += norm.logpdf(y, 0, 1.0).sum()
            else:
                # Marginal likelihood integrating out weights
                wp = self.weight_posteriors[graph_idx].get(var)
                if wp is not None:
                    ll += wp.log_marginal_likelihood(X_pa, y)
                else:
                    ll += norm.logpdf(y, 0, np.sqrt(self.sigma_eps2)).sum()
        return ll

    def update(self, intv_data: np.ndarray, target_var: str,
               all_data: np.ndarray = None) -> np.ndarray:
        """Bayesian update over graphs and weights.

        1. Update weight posteriors for each graph given all accumulated data
        2. Compute marginal likelihood for each graph on interventional data
        3. Update graph belief: posterior proportional to marginal_lik * prior
        """
        est_data = all_data if all_data is not None else intv_data

        log_liks = np.zeros(self.K)
        for k in range(self.K):
            self.update_weight_posteriors(k, est_data, target_var)
            log_liks[k] = self.compute_log_marginal_likelihood(k, intv_data, target_var)

        log_post = log_liks + np.log(self.belief + 1e-300)
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        self.belief = post
        self.belief_history.append(post.copy())
        return post

    def get_weight_posterior_summary(self, graph_idx: int) -> dict:
        """Posterior mean, std, and 95% CI for all edge weights under a graph."""
        summary = {}
        for var in self.variable_order:
            parents = self.get_parents(graph_idx, var)
            if not parents:
                continue
            wp = self.weight_posteriors[graph_idx].get(var)
            if wp is None:
                continue
            cov = wp.covariance
            for i, parent in enumerate(parents):
                edge_key = f"w_{parent}{var}"
                std = float(np.sqrt(cov[i, i]))
                summary[edge_key] = {
                    "mean": float(wp.mean[i]),
                    "std": std,
                    "lower_95": float(wp.mean[i] - 1.96 * std),
                    "upper_95": float(wp.mean[i] + 1.96 * std),
                }
        return summary

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
            "weight_posteriors": self.get_weight_posterior_summary(mi),
        }


if __name__ == "__main__":
    gb = GraphBelief(tau=3.0, sigma_w2=0.5, sigma_eps2=0.3)
    print("Prior:", gb.prior.round(4))
    print(f"Weight prior: N(0, {gb.sigma_w2}) for all edges (uniform, zero-mean)")
    for k, c in enumerate(gb.candidates):
        print(f"\n  {c['id']}: conf={c['confidence']}, prior={gb.prior[k]:.4f}")
        wp_summary = gb.get_weight_posterior_summary(k)
        for edge, stats in wp_summary.items():
            print(f"    {edge}: mean={stats['mean']:.3f}, std={stats['std']:.3f} "
                  f"[{stats['lower_95']:.3f}, {stats['upper_95']:.3f}]")

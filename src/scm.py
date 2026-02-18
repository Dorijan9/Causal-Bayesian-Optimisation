"""
Structural Causal Model (SCM) for the KEGG-grounded causal graph.

Supports:
- Observational sampling (ancestral sampling in topological order)
- Interventional sampling via do-calculus (hard interventions)
- Linear Gaussian SCM with excitatory and inhibitory edges
"""

import json
import numpy as np
from typing import Optional


class LinearGaussianSCM:
    """Linear Gaussian Structural Causal Model."""

    def __init__(self, dag_path: str = "data/ground_truth_dag.json"):
        with open(dag_path) as f:
            self.dag_data = json.load(f)

        self.variables = list(self.dag_data["variables"].keys())
        self.n_vars = len(self.variables)
        self.var_to_idx = {v: i for i, v in enumerate(self.variables)}
        self.topological_order = self.dag_data["topological_order"]

        # Build weight matrix W where W[i,j] = weight of edge i -> j
        params = self.dag_data["scm_parameters"]
        self.noise_var = params["noise_variance"]
        self.root_var = 1.0

        self.W = np.zeros((self.n_vars, self.n_vars))
        self.edge_signs = {}
        for edge in self.dag_data["edges"]:
            i = self.var_to_idx[edge["source"]]
            j = self.var_to_idx[edge["target"]]
            self.W[i, j] = edge["weight"]
            self.edge_signs[(edge["source"], edge["target"])] = edge["sign"]

        # Binary adjacency
        self.adj = (self.W != 0).astype(int)

    def get_parents(self, var: str) -> list:
        """Get parent variables of a given variable."""
        j = self.var_to_idx[var]
        return [v for i, v in enumerate(self.variables) if self.W[i, j] != 0]

    def sample_observational(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample n observations via ancestral sampling. Returns (n, n_vars)."""
        rng = np.random.default_rng(seed)
        data = np.zeros((n, self.n_vars))

        for var in self.topological_order:
            j = self.var_to_idx[var]
            parents = self.get_parents(var)

            if len(parents) == 0:
                data[:, j] = rng.normal(0, np.sqrt(self.root_var), n)
            else:
                mean = sum(self.W[self.var_to_idx[p], j] * data[:, self.var_to_idx[p]] for p in parents)
                data[:, j] = mean + rng.normal(0, np.sqrt(self.noise_var), n)

        return data

    def sample_interventional(self, target_var: str, intervention_value: float,
                               n: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample n observations under do(target_var = intervention_value). Returns (n, n_vars)."""
        rng = np.random.default_rng(seed)
        data = np.zeros((n, self.n_vars))

        for var in self.topological_order:
            j = self.var_to_idx[var]

            if var == target_var:
                data[:, j] = intervention_value
                continue

            parents = self.get_parents(var)
            if len(parents) == 0:
                data[:, j] = rng.normal(0, np.sqrt(self.root_var), n)
            else:
                mean = sum(self.W[self.var_to_idx[p], j] * data[:, self.var_to_idx[p]] for p in parents)
                data[:, j] = mean + rng.normal(0, np.sqrt(self.noise_var), n)

        return data


if __name__ == "__main__":
    scm = LinearGaussianSCM()
    print("Variables:", scm.variables)
    print("Topological order:", scm.topological_order)
    print("\nWeight matrix:\n", scm.W)
    for v in scm.variables:
        print(f"Parents of {v}: {scm.get_parents(v)}")

    obs = scm.sample_observational(1000, seed=42)
    print(f"\nObservational means: {obs.mean(0).round(3)}")
    print(f"Observational stds:  {obs.std(0).round(3)}")

    for t in scm.variables:
        intv = scm.sample_interventional(t, 2.0, 1000, seed=42)
        print(f"do({t}=2.0) means: {intv.mean(0).round(3)}")

# CBRA: Causal Bayesian Reasoning Agent

Iterative causal graph discovery via Bayesian optimisation with KEGG-grounded pathway priors and Bayesian edge weight inference.

## Overview

CBRA performs two-level Bayesian inference:

1. **Graph level:** Posterior over candidate DAG structures, with LLM-derived structural priors updated via marginal likelihood (weights integrated out).
2. **Weight level:** Posterior over causal effect strengths for each candidate graph, starting from a uniform zero-mean Gaussian prior `w_ij ~ N(0, σ²_w)` and updated via Bayesian linear regression.

The LLM contributes *structural* knowledge (which edges exist); *parametric* knowledge (sign, magnitude) comes entirely from interventional data.

## Ground-Truth DAG

Five phenotypic variables grounded to KEGG pathways:

```
E (PI3K-Akt, hsa04151)
│  w*_ED = +0.75
▼
D (HIF-1, hsa04066)
├──────────────────┐
│  w*_DA = +0.80   │  w*_DC = -0.50 (inhibitory)
▼                  ▼
A (Glycolysis)     C (OXPHOS, hsa00190)
│  w*_AB = +0.70   ▲
▼                  │  w*_BC = +0.60
B (TCA, hsa00020)──┘
```

Ground-truth weights are unknown to the agent. All weight priors start at `N(0, σ²_w)`.

## Algorithm

```
1. Set weight priors: w_ij ~ N(0, σ²_w) for all candidate edges
2. Prompt LLM for K candidate graphs with confidence scores
3. Set graph prior: b_0 = softmax(τ · [c_1, ..., c_K])
4. For t = 1, ..., T:
   a. Select intervention via Expected Information Gain (EIG)
   b. Collect interventional data
   c. For each candidate graph:
      - Update weight posteriors (Bayesian linear regression)
      - Compute marginal likelihood (weights integrated out)
   d. Update graph posterior: b_t ∝ P(D_t | G_k) · b_{t-1}
   e. If converged (max(b_t) > 0.95): break
5. Return graph posterior, weight posteriors, MAP estimate
```

## Key Features

- **Zero-mean weight priors**: Agent starts agnostic about excitatory vs. inhibitory effects
- **Marginal likelihood**: Integrates out weights analytically (linear Gaussian closed form)
- **Bayesian linear regression**: Posterior precision, mean, covariance per edge per graph
- **Weight uncertainty propagation**: Multiplicative variance growth through causal chains
- **Weight recovery metrics**: RMSE and 95% CI coverage alongside structural metrics (SHD, F1)

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| SHD | Structural Hamming Distance |
| Edge F1 | Precision/recall of directed edges |
| Weight RMSE | Distance between posterior means and true weights |
| Weight Coverage | Fraction of true weights within 95% credible intervals |
| Entropy Reduction | H[b_0] - H[b_T] |

## Usage

```bash
pip install numpy scipy matplotlib
python -m src.run_experiment
```

## Project Structure

```
CBO/
├── configs/experiment_config.json    # Hyperparameters (σ²_w, σ²_ε, τ, etc.)
├── data/
│   ├── ground_truth_dag.json         # KEGG-grounded DAG with edge provenance
│   └── candidate_graphs.json         # 5 candidate DAGs with LLM confidences
├── src/
│   ├── scm.py                        # Linear Gaussian SCM (observational + interventional)
│   ├── graph_belief.py               # Bayesian inference: weight posteriors + graph posterior
│   ├── acquisition.py                # EIG-based intervention selection
│   ├── metrics.py                    # SHD, F1, weight RMSE, weight coverage
│   ├── run_experiment.py             # Main CBO loop + baselines
│   └── plot_results.py               # Visualisation utilities
├── logs/                             # Experiment results (JSON + CSV)
└── plots/                            # Generated figures
```

## Baselines

| Method | Description |
|--------|-------------|
| Uniform prior CBO | CBO with b_0 = uniform (no LLM prior) |
| Random intervention | Random targets instead of EIG |
| Single-shot LLM | Highest-confidence graph, no data |

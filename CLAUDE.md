# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CBRA (Causal Bayesian Reasoning Agent) is an academic research project implementing **Causal Bayesian Optimization with Bayesian Edge Weight Inference**. It combines LLM-derived graph structure priors with Bayesian weight estimation from interventional data to identify causal graphs in a KEGG-grounded metabolic network (5 variables: E=PI3K-Akt, D=HIF-1, A=Glycolysis, B=TCA, C=OXPHOS).

## Commands

**Install dependencies:**
```bash
pip install numpy>=1.21 scipy>=1.7 matplotlib>=3.5
```

**Run the full experiment (main CBO loop + all baselines):**
```bash
python -m src.run_experiment
```

**Generate plots from saved results:**
```bash
python src/plot_results.py logs/experiment_results_<timestamp>.json
```

**Test individual modules (each has a `__main__` block):**
```bash
python src/scm.py           # Test SCM sampling
python src/graph_belief.py  # Test Bayesian graph/weight inference
python src/acquisition.py   # Test EIG computation
python src/metrics.py       # Test evaluation metrics
```

## Architecture

The CBO loop in `run_experiment.py:run_cbo()` orchestrates these modules each iteration:

1. **`src/scm.py` — `LinearGaussianSCM`**: Ground-truth causal model. Performs ancestral sampling (`sample_observational`) and hard interventions via do-calculus (`sample_interventional`). Loaded from `data/ground_truth_dag.json`.

2. **`src/graph_belief.py` — `GraphBelief` + `WeightPosterior`**: Two-level Bayesian inference:
   - `WeightPosterior`: Bayesian linear regression with zero-mean Gaussian priors; computes closed-form posteriors and **log marginal likelihood** (weights analytically integrated out).
   - `GraphBelief`: Loads 5 LLM candidate DAGs from `data/candidate_graphs.json` with confidence scores. Graph posterior updated via `b_t ∝ P(D_t | G_k) · b_{t-1}` using marginal likelihoods. Temperature parameter `τ=3.0` sharpens LLM prior via softmax.

3. **`src/acquisition.py` — `expected_information_gain()`**: Selects the intervention target maximizing expected entropy reduction. Simulates outcomes under current graph+weight belief using Monte Carlo (`n_eig_simulations=200`).

4. **`src/metrics.py`**: Structural metrics (SHD, F1/precision/recall on directed edges) and weight recovery metrics (RMSE of posterior means, 95% CI coverage of true weights).

5. **`src/plot_results.py`**: Six visualization functions covering posterior evolution, entropy reduction, EIG heatmaps, weight posteriors, and baseline comparison. Outputs to `logs/plots/`.

## Configuration

All hyperparameters are in `configs/experiment_config.json`:
- `scm.observation_noise_variance`: 0.3
- `weight_prior.variance`: 0.5 (zero-mean prior)
- `graph_prior.temperature_tau`: 3.0
- `cbo.max_iterations`: 4, `convergence_threshold`: 0.95
- `cbo.n_eig_simulations`: 200

## Data

- `data/ground_truth_dag.json`: 5-node DAG (E→D→A→B→C, D→C inhibitory) with true edge weights
- `data/candidate_graphs.json`: 5 candidate DAGs with LLM confidence scores (G1 is the correct structure)

## Outputs

Results are timestamped and saved to `logs/`:
- `experiment_results_<timestamp>.json`: Full iteration history, metrics, baselines
- `cbo_iterations_<timestamp>.csv`: Per-iteration tabular log (target, MAP graph, entropy, SHD, F1, weight RMSE, coverage)

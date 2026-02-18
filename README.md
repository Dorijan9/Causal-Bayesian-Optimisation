# CBRA: Causal Bayesian Reasoning Agent

**Bayesian Optimisation over a Structured Intervention Space with LLM-Informed Graph Priors**

## Overview

CBRA performs iterative causal graph discovery using Causal Bayesian Optimisation (CBO). Given a set of phenotypic variables grounded in the KEGG pathway database, the agent:

1. Constructs a **ground-truth causal DAG** from KEGG pathway cross-references
2. Elicits **candidate causal graphs** from an LLM with confidence-weighted priors
3. Selects **maximally informative interventions** via Expected Information Gain (EIG)
4. Performs **Bayesian posterior updates** over candidate graphs using interventional data
5. Recovers the true causal structure in 3-4 iterations

## Ground-Truth DAG (KEGG-Grounded)

```
E (PI3K-Akt, hsa04151)
  |
  v
D (HIF-1, hsa04066) ------+
  |                        | (inhibitory)
  v                        v
A (Glycolysis, hsa00010)   C (OXPHOS, hsa00190)
  |                        ^
  v                        |
B (TCA cycle, hsa00020) ---+
```

Every edge is justified by documented KEGG pathway cross-references. See `data/ground_truth_dag.json` for full provenance.

## Project Structure

```
CBO/
├── configs/
│   └── experiment_config.json    # All hyperparameters
├── data/
│   ├── ground_truth_dag.json     # KEGG-grounded DAG with edge justifications
│   ├── candidate_graphs.json     # K=5 LLM-proposed candidate DAGs
│   ├── kegg_adjacency.json       # Pathway adjacency with shared genes
│   └── llm_prompt_config.json    # LLM prompting template
├── src/
│   ├── scm.py                    # Linear Gaussian SCM
│   ├── graph_belief.py           # Prior, likelihood, Bayesian update
│   ├── acquisition.py            # Expected Information Gain
│   ├── metrics.py                # SHD, precision, recall, F1
│   ├── run_experiment.py         # Main CBO loop + baselines
│   └── plot_results.py           # Visualisation utilities
├── logs/                         # Experiment results
├── plots/                        # Generated figures
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.run_experiment
```

This runs the main CBO experiment plus three baselines (uniform prior, random intervention, single-shot LLM).

## Generate Plots

```bash
python -m src.plot_results logs/experiment_results_XXXXXXXX.json
```

## License

MIT

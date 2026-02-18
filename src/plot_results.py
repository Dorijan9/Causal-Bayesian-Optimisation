"""
Plotting utilities for CBRA experiment results.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def plot_posterior_evolution(results: dict, save_path: str = "plots/posterior_evolution.png"):
    """Plot how posterior belief over graphs evolves across CBO iterations."""
    iterations = results["iterations"]
    K = len(iterations[0]["belief_after"])

    fig, ax = plt.subplots(figsize=(10, 6))

    graph_ids = [f"G{k+1}" for k in range(K)]
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    # Include initial belief
    all_beliefs = [results["initial_belief"]] + [it["belief_after"] for it in iterations]
    x = list(range(len(all_beliefs)))
    x_labels = ["Prior"] + [f"Iter {it['iteration']}" for it in iterations]

    for k in range(K):
        y = [b[k] for b in all_beliefs]
        ax.plot(x, y, "o-", color=colors[k], label=graph_ids[k], linewidth=2, markersize=8)

    ax.set_xlabel("CBO Iteration", fontsize=13)
    ax.set_ylabel("Posterior Probability", fontsize=13)
    ax.set_title("Posterior Evolution Over Candidate Graphs", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate intervention targets
    for i, it in enumerate(iterations):
        ax.annotate(
            f"do({it['target_variable']})",
            xy=(i + 1, max(it["belief_after"]) + 0.03),
            ha="center", fontsize=9, color="gray",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_entropy_reduction(results: dict, save_path: str = "plots/entropy_reduction.png"):
    """Plot entropy reduction across iterations."""
    iterations = results["iterations"]

    entropies = [results["initial_entropy"]] + [it["entropy_after"] for it in iterations]
    x = list(range(len(entropies)))
    x_labels = ["Prior"] + [f"Iter {it['iteration']}" for it in iterations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, entropies, "s-", color="#2E86C1", linewidth=2, markersize=10)
    ax.fill_between(x, entropies, alpha=0.15, color="#2E86C1")

    ax.set_xlabel("CBO Iteration", fontsize=13)
    ax.set_ylabel("Belief Entropy (nats)", fontsize=13)
    ax.set_title("Entropy Reduction Across CBO Iterations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_eig_heatmap(results: dict, save_path: str = "plots/eig_heatmap.png"):
    """Plot EIG scores as a heatmap across iterations and variables."""
    iterations = results["iterations"]
    variables = sorted(iterations[0]["eig_scores"].keys())
    n_iter = len(iterations)

    eig_matrix = np.zeros((n_iter, len(variables)))
    for t, it in enumerate(iterations):
        for j, var in enumerate(variables):
            eig_matrix[t, j] = it["eig_scores"][var]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(eig_matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels([f"do({v})" for v in variables], fontsize=11)
    ax.set_yticks(range(n_iter))
    ax.set_yticklabels([f"Iter {it['iteration']}" for it in iterations], fontsize=11)
    ax.set_title("Expected Information Gain per Intervention", fontsize=14)

    # Annotate selected interventions
    for t, it in enumerate(iterations):
        j = variables.index(it["target_variable"])
        ax.add_patch(plt.Rectangle((j - 0.5, t - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))

    plt.colorbar(im, ax=ax, label="EIG (nats)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_shd_comparison(main_results: dict, uniform_results: dict,
                        random_results: dict, llm_results: dict,
                        save_path: str = "plots/baseline_comparison.png"):
    """Bar chart comparing SHD across methods."""
    methods = ["LLM+CBO\n(EIG)", "Uniform Prior\n(CBO)", "Random\nIntervention", "Single-Shot\nLLM"]
    shds = [
        main_results["final"]["shd"],
        uniform_results["final"]["shd"],
        random_results["mean_shd"],
        llm_results["shd"],
    ]
    colors = ["#27AE60", "#2E86C1", "#E74C3C", "#F39C12"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, shds, color=colors, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Structural Hamming Distance (SHD)", fontsize=13)
    ax.set_title("Graph Recovery: SHD Comparison Across Methods", fontsize=14)
    ax.set_ylim(0, max(shds) + 1.5)

    for bar, shd in zip(bars, shds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{shd:.1f}", ha="center", fontsize=12, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_dag(adj: np.ndarray, variable_names: list,
             title: str = "Causal DAG", save_path: str = "plots/dag.png"):
    """Simple DAG visualisation using matplotlib."""
    n = len(variable_names)

    # Layout positions (manually arranged for our 5-variable graph)
    positions = {
        "E": (0.5, 1.0),
        "D": (0.5, 0.7),
        "A": (0.2, 0.4),
        "C": (0.8, 0.1),
        "B": (0.2, 0.1),
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                src = variable_names[i]
                tgt = variable_names[j]
                x0, y0 = positions[src]
                x1, y1 = positions[tgt]

                color = "#E74C3C" if adj[i, j] < 0 else "#2C3E50"
                style = "dashed" if adj[i, j] < 0 else "solid"

                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=2,
                        linestyle=style, shrinkA=20, shrinkB=20,
                    ),
                )

    # Draw nodes
    for var, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.06, color="#3498DB", ec="black", lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, var, ha="center", va="center", fontsize=16,
                fontweight="bold", color="white", zorder=6)

    # Add KEGG labels
    kegg_labels = {
        "A": "Glycolysis\nhsa00010",
        "B": "TCA cycle\nhsa00020",
        "C": "OXPHOS\nhsa00190",
        "D": "HIF-1\nhsa04066",
        "E": "PI3K-Akt\nhsa04151",
    }
    for var, (x, y) in positions.items():
        ax.text(x, y - 0.1, kegg_labels[var], ha="center", va="top",
                fontsize=9, color="gray")

    # Legend
    exc_patch = mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#2C3E50", lw=2)
    inh_patch = mpatches.FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#E74C3C", lw=2, linestyle="dashed")
    ax.legend(
        [plt.Line2D([0], [0], color="#2C3E50", lw=2),
         plt.Line2D([0], [0], color="#E74C3C", lw=2, linestyle="dashed")],
        ["Excitatory", "Inhibitory"],
        fontsize=11, loc="lower right",
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(results_path: str):
    """Generate all plots from a saved results JSON."""
    with open(results_path) as f:
        all_results = json.load(f)

    Path("plots").mkdir(exist_ok=True)

    plot_posterior_evolution(all_results["main_cbo"])
    plot_entropy_reduction(all_results["main_cbo"])
    plot_eig_heatmap(all_results["main_cbo"])

    if all_results.get("baseline_uniform_prior"):
        plot_shd_comparison(
            all_results["main_cbo"],
            all_results["baseline_uniform_prior"],
            all_results["baseline_random_intervention"],
            all_results["baseline_single_shot_llm"],
        )

    # Plot ground truth DAG
    with open("data/ground_truth_dag.json") as f:
        gt = json.load(f)

    W = np.array([[gt["scm_parameters"]["weights"].get(
        f"w_{gt['adjacency_matrix']['order'][i]}{gt['adjacency_matrix']['order'][j]}", 0)
        for j in range(5)] for i in range(5)])

    # Use the adjacency with sign info from edges
    adj_signed = np.zeros((5, 5))
    var_order = gt["adjacency_matrix"]["order"]
    for edge in gt["edges"]:
        i = var_order.index(edge["source"])
        j = var_order.index(edge["target"])
        adj_signed[i, j] = edge["weight"]

    plot_dag(adj_signed, var_order, "Ground-Truth DAG (KEGG-grounded)", "plots/ground_truth_dag.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_all_plots(sys.argv[1])
    else:
        # Just plot the ground truth DAG
        Path("plots").mkdir(exist_ok=True)
        with open("data/ground_truth_dag.json") as f:
            gt = json.load(f)
        var_order = gt["adjacency_matrix"]["order"]
        adj_signed = np.zeros((5, 5))
        for edge in gt["edges"]:
            i = var_order.index(edge["source"])
            j = var_order.index(edge["target"])
            adj_signed[i, j] = edge["weight"]
        plot_dag(adj_signed, var_order, "Ground-Truth DAG (KEGG-grounded)", "plots/ground_truth_dag.png")

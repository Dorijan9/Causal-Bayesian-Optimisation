"""
CBRA: Causal Bayesian Reasoning Agent - Main Experiment Runner

Runs the full CBO loop with Bayesian edge weight inference:
1. Load ground-truth DAG and SCM
2. Load candidate graphs, construct graph prior, init zero-mean weight priors
3. Iteratively: select intervention via EIG, collect data, update weight
   posteriors and graph posterior
4. Evaluate structure recovery (SHD, F1) and weight recovery (RMSE, coverage)
5. Run baselines for comparison
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief
from src.acquisition import select_intervention, random_intervention, entropy
from src.metrics import evaluate_graph, evaluate_weights


# Ground-truth weight keys matching edge naming convention
TRUE_WEIGHTS = {
    "w_ED": 0.75,
    "w_DA": 0.80,
    "w_AB": 0.70,
    "w_BC": 0.60,
    "w_DC": -0.50,
}


def run_cbo(
    scm: LinearGaussianSCM,
    belief: GraphBelief,
    config: dict,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run the full CBO loop with Bayesian weight inference."""
    rng = np.random.default_rng(seed)
    cbo_cfg = config["cbo"]
    scm_cfg = config["scm"]

    max_iter = cbo_cfg["max_iterations"]
    threshold = cbo_cfg["convergence_threshold"]
    n_sim = cbo_cfg["n_eig_simulations"]
    n_samples = scm_cfg["n_interventional_samples_per_iter"]
    intv_val = scm_cfg["intervention_value"]

    # Collect optional observational data
    obs_data = scm.sample_observational(scm_cfg["n_observational_samples"], seed=seed)

    results = {
        "iterations": [],
        "initial_belief": belief.belief.tolist(),
        "initial_entropy": float(entropy(belief.belief)),
    }

    all_intv_data = []

    if verbose:
        print("=" * 70)
        print("CBRA: Causal Bayesian Optimisation with Bayesian Edge Weights")
        print("=" * 70)
        print(f"Weight prior: N(0, {belief.sigma_w2}) for all edges (uniform zero-mean)")
        print(f"Obs. noise: sigma_eps^2 = {belief.sigma_eps2}")
        print(f"Initial belief: {belief.belief.round(4)}")
        print(f"Initial entropy: {entropy(belief.belief):.4f}")
        print()

    for t in range(1, max_iter + 1):
        if verbose:
            print(f"--- Iteration {t} ---")

        # Step 1: Select intervention via EIG
        iter_seed = rng.integers(0, 2**31)
        target, eig_scores = select_intervention(
            scm, belief, intv_val, n_sim, n_samples, seed=iter_seed
        )

        if verbose:
            print(f"EIG scores: { {k: round(v, 4) for k, v in sorted(eig_scores.items(), key=lambda x: -x[1])} }")
            print(f"Selected intervention: do({target} = {intv_val})")

        # Step 2: Perform intervention and collect data
        intv_data = scm.sample_interventional(
            target, intv_val, n_samples, seed=iter_seed + 1
        )
        all_intv_data.append(intv_data)

        # Combine all data for weight estimation
        combined_data = np.vstack([obs_data] + all_intv_data)

        # Step 3: Update graph posterior and weight posteriors
        old_belief = belief.belief.copy()
        new_belief = belief.update(intv_data, target, combined_data)

        # Step 4: Evaluate structure
        map_idx = belief.map_estimate()
        map_graph = belief.candidates[map_idx]
        gt_adj = np.array(scm.adj)
        map_adj = map_graph["adjacency"]
        eval_struct = evaluate_graph(gt_adj, map_adj)

        # Step 5: Evaluate weight recovery for MAP graph
        wp_summary = belief.get_weight_posterior_summary(map_idx)
        eval_wt = evaluate_weights(wp_summary, TRUE_WEIGHTS)

        iter_result = {
            "iteration": t,
            "target_variable": target,
            "eig_scores": eig_scores,
            "belief_before": old_belief.tolist(),
            "belief_after": new_belief.tolist(),
            "entropy_before": float(entropy(old_belief)),
            "entropy_after": float(entropy(new_belief)),
            "map_graph": map_graph["id"],
            "map_probability": float(new_belief[map_idx]),
            "shd": eval_struct["shd"],
            "precision": eval_struct["precision"],
            "recall": eval_struct["recall"],
            "f1": eval_struct["f1"],
            "weight_rmse": eval_wt["weight_rmse"],
            "weight_coverage": eval_wt["weight_coverage"],
            "weight_posteriors": wp_summary,
            "converged": belief.has_converged(threshold),
        }
        results["iterations"].append(iter_result)

        if verbose:
            print(f"Posterior: {new_belief.round(4)}")
            print(f"MAP graph: {map_graph['id']} (P={new_belief[map_idx]:.4f})")
            print(f"Entropy: {entropy(old_belief):.4f} -> {entropy(new_belief):.4f}")
            print(f"SHD: {eval_struct['shd']}, F1: {eval_struct['f1']:.3f}")
            print(f"Weight RMSE: {eval_wt['weight_rmse']:.4f}, Coverage: {eval_wt['weight_coverage']:.2%}")
            if wp_summary:
                print("Weight posteriors (MAP graph):")
                for edge, stats in sorted(wp_summary.items()):
                    true_w = TRUE_WEIGHTS.get(edge, "?")
                    print(f"  {edge}: mean={stats['mean']:+.3f} ± {stats['std']:.3f} "
                          f"[{stats['lower_95']:+.3f}, {stats['upper_95']:+.3f}]  (true={true_w})")
            print()

        if belief.has_converged(threshold):
            if verbose:
                print(f"*** Converged at iteration {t}! ***")
            break

    # Final summary
    final_map_idx = belief.map_estimate()
    final_map = belief.candidates[final_map_idx]
    final_eval = evaluate_graph(gt_adj, final_map["adjacency"])
    final_wp = belief.get_weight_posterior_summary(final_map_idx)
    final_wt_eval = evaluate_weights(final_wp, TRUE_WEIGHTS)

    results["final"] = {
        "map_graph": final_map["id"],
        "map_probability": float(belief.belief[final_map_idx]),
        "final_entropy": float(entropy(belief.belief)),
        "entropy_reduction": float(entropy(belief.prior) - entropy(belief.belief)),
        "total_iterations": len(results["iterations"]),
        "converged": belief.has_converged(threshold),
        "weight_posteriors": final_wp,
        **final_eval,
        **final_wt_eval,
    }

    if verbose:
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"MAP graph: {final_map['id']}")
        print(f"MAP probability: {belief.belief[final_map_idx]:.4f}")
        print(f"SHD: {final_eval['shd']}")
        print(f"Precision: {final_eval['precision']:.3f}")
        print(f"Recall: {final_eval['recall']:.3f}")
        print(f"F1: {final_eval['f1']:.3f}")
        print(f"Weight RMSE: {final_wt_eval['weight_rmse']:.4f}")
        print(f"Weight coverage (95% CI): {final_wt_eval['weight_coverage']:.2%}")
        print(f"Entropy reduction: {results['final']['entropy_reduction']:.4f}")
        correct = final_map["id"] == "G1"
        print(f"Correct graph recovered: {correct}")

    return results


def run_baseline_uniform_prior(scm, config, seed=42, verbose=True):
    """Baseline: CBO with uniform graph prior (tau=0)."""
    if verbose:
        print("\n>>> BASELINE: Uniform Prior CBO <<<")
    wp_cfg = config["weight_prior"]
    belief = GraphBelief(tau=0.0, sigma_w2=wp_cfg["variance"],
                         sigma_eps2=config["scm"]["observation_noise_variance"])
    return run_cbo(scm, belief, config, seed, verbose)


def run_baseline_random_intervention(scm, config, n_repeats=50, seed=42, verbose=True):
    """Baseline: random intervention selection."""
    if verbose:
        print("\n>>> BASELINE: Random Intervention <<<")

    rng = np.random.default_rng(seed)
    wp_cfg = config["weight_prior"]
    all_results = []

    for rep in range(n_repeats):
        belief = GraphBelief(
            tau=config["graph_prior"]["temperature_tau"],
            sigma_w2=wp_cfg["variance"],
            sigma_eps2=config["scm"]["observation_noise_variance"],
        )
        scm_cfg = config["scm"]
        n_samples = scm_cfg["n_interventional_samples_per_iter"]
        intv_val = scm_cfg["intervention_value"]
        obs_data = scm.sample_observational(scm_cfg["n_observational_samples"], seed=seed + rep)
        all_intv = []

        for t in range(config["cbo"]["max_iterations"]):
            target = random_intervention(scm, seed=rng.integers(0, 2**31))
            intv_data = scm.sample_interventional(target, intv_val, n_samples,
                                                   seed=rng.integers(0, 2**31))
            all_intv.append(intv_data)
            combined = np.vstack([obs_data] + all_intv)
            belief.update(intv_data, target, combined)

        map_idx = belief.map_estimate()
        map_adj = belief.candidates[map_idx]["adjacency"]
        gt_adj = np.array(scm.adj)
        eval_r = evaluate_graph(gt_adj, map_adj)

        wp_summary = belief.get_weight_posterior_summary(map_idx)
        wt_eval = evaluate_weights(wp_summary, TRUE_WEIGHTS)

        all_results.append({
            "map_graph": belief.candidates[map_idx]["id"],
            "map_prob": float(belief.belief[map_idx]),
            **eval_r,
            **wt_eval,
        })

    # Aggregate
    shds = [r["shd"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    w_rmses = [r["weight_rmse"] for r in all_results]
    w_covs = [r["weight_coverage"] for r in all_results]
    correct = sum(1 for r in all_results if r["map_graph"] == "G1")

    summary = {
        "n_repeats": n_repeats,
        "mean_shd": float(np.mean(shds)),
        "std_shd": float(np.std(shds)),
        "mean_f1": float(np.mean(f1s)),
        "mean_weight_rmse": float(np.mean(w_rmses)),
        "mean_weight_coverage": float(np.mean(w_covs)),
        "correct_recovery_rate": correct / n_repeats,
    }

    if verbose:
        print(f"  Mean SHD: {summary['mean_shd']:.2f} +/- {summary['std_shd']:.2f}")
        print(f"  Mean F1: {summary['mean_f1']:.3f}")
        print(f"  Mean Weight RMSE: {summary['mean_weight_rmse']:.4f}")
        print(f"  Mean Weight Coverage: {summary['mean_weight_coverage']:.2%}")
        print(f"  Correct recovery rate: {summary['correct_recovery_rate']:.2%}")

    return summary


def run_baseline_single_shot_llm(config, verbose=True):
    """Baseline: just take the highest-confidence LLM graph, no data."""
    if verbose:
        print("\n>>> BASELINE: Single-Shot LLM <<<")

    wp_cfg = config["weight_prior"]
    belief = GraphBelief(
        tau=config["graph_prior"]["temperature_tau"],
        sigma_w2=wp_cfg["variance"],
        sigma_eps2=config["scm"]["observation_noise_variance"],
    )
    map_idx = belief.map_estimate()
    map_graph = belief.candidates[map_idx]

    with open("data/ground_truth_dag.json") as f:
        gt = json.load(f)
    gt_adj = np.array(gt["adjacency_matrix"]["matrix"])
    eval_r = evaluate_graph(gt_adj, np.array(map_graph["adjacency"]))

    # No data -> weight posteriors are just the prior (zero-mean)
    wp_summary = belief.get_weight_posterior_summary(map_idx)
    wt_eval = evaluate_weights(wp_summary, TRUE_WEIGHTS)

    result = {
        "map_graph": map_graph["id"],
        "map_probability": float(belief.prior[map_idx]),
        **eval_r,
        **wt_eval,
    }

    if verbose:
        print(f"  Selected graph: {map_graph['id']} (prior P={belief.prior[map_idx]:.4f})")
        print(f"  SHD: {eval_r['shd']}, F1: {eval_r['f1']:.3f}")
        print(f"  Weight RMSE: {wt_eval['weight_rmse']:.4f} (no data, prior only)")
        print(f"  Weight Coverage: {wt_eval['weight_coverage']:.2%}")

    return result


def main():
    """Run the full experiment."""
    with open("configs/experiment_config.json") as f:
        config = json.load(f)

    seed = config["random_seed"]
    wp_cfg = config["weight_prior"]
    sigma_eps2 = config["scm"]["observation_noise_variance"]

    # Initialise SCM and belief
    scm = LinearGaussianSCM()
    belief = GraphBelief(
        tau=config["graph_prior"]["temperature_tau"],
        sigma_w2=wp_cfg["variance"],
        sigma_eps2=sigma_eps2,
    )

    # Run main CBO
    print("\n" + "=" * 70)
    print("MAIN EXPERIMENT: LLM-Prior CBO with Bayesian Edge Weights")
    print("=" * 70)
    main_results = run_cbo(scm, belief, config, seed=seed, verbose=True)

    # Run baselines
    uniform_results = run_baseline_uniform_prior(scm, config, seed=seed, verbose=True)
    random_results = run_baseline_random_intervention(
        scm, config, n_repeats=config["baselines"]["random_intervention"]["n_repeats"],
        seed=seed, verbose=True
    )
    llm_results = run_baseline_single_shot_llm(config, verbose=True)

    # Save results
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {
        "timestamp": timestamp,
        "config": config,
        "main_cbo": main_results,
        "baseline_uniform_prior": uniform_results,
        "baseline_random_intervention": random_results,
        "baseline_single_shot_llm": llm_results,
    }

    results_path = f"logs/experiment_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save CSV log
    csv_path = f"logs/cbo_iterations_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iteration", "target_variable", "map_graph", "map_probability",
            "entropy_before", "entropy_after", "shd", "precision", "recall", "f1",
            "weight_rmse", "weight_coverage",
        ])
        writer.writeheader()
        for it in main_results["iterations"]:
            writer.writerow({k: it[k] for k in writer.fieldnames})
    print(f"Iteration log saved to {csv_path}")


if __name__ == "__main__":
    main()

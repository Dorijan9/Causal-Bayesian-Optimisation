"""
Evaluation metrics for causal graph recovery.
"""

import numpy as np


def structural_hamming_distance(adj_true: np.ndarray, adj_pred: np.ndarray) -> int:
    """
    Structural Hamming Distance (SHD).

    Counts the number of edge additions, deletions, and reversals needed
    to transform adj_pred into adj_true.
    """
    n = adj_true.shape[0]
    shd = 0

    for i in range(n):
        for j in range(i + 1, n):
            true_ij = adj_true[i, j] != 0
            true_ji = adj_true[j, i] != 0
            pred_ij = adj_pred[i, j] != 0
            pred_ji = adj_pred[j, i] != 0

            if true_ij and not true_ji:
                # True edge i->j
                if not pred_ij and not pred_ji:
                    shd += 1  # Missing edge
                elif pred_ji and not pred_ij:
                    shd += 1  # Reversed edge
                elif pred_ij and pred_ji:
                    shd += 1  # Extra reverse edge
            elif true_ji and not true_ij:
                # True edge j->i
                if not pred_ij and not pred_ji:
                    shd += 1
                elif pred_ij and not pred_ji:
                    shd += 1
                elif pred_ij and pred_ji:
                    shd += 1
            elif not true_ij and not true_ji:
                # No true edge
                if pred_ij:
                    shd += 1
                if pred_ji:
                    shd += 1
            elif true_ij and true_ji:
                # Bidirectional true edge (shouldn't happen in DAG)
                if not pred_ij:
                    shd += 1
                if not pred_ji:
                    shd += 1

    return shd


def edge_precision_recall_f1(
    adj_true: np.ndarray, adj_pred: np.ndarray
) -> dict:
    """
    Compute edge-level precision, recall, and F1.

    Considers directed edges: (i,j) in pred vs (i,j) in true.
    """
    true_edges = set(zip(*np.where(adj_true != 0)))
    pred_edges = set(zip(*np.where(adj_pred != 0)))

    if len(pred_edges) == 0:
        precision = 0.0
    else:
        precision = len(true_edges & pred_edges) / len(pred_edges)

    if len(true_edges) == 0:
        recall = 0.0
    else:
        recall = len(true_edges & pred_edges) / len(true_edges)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": len(true_edges & pred_edges),
        "false_positives": len(pred_edges - true_edges),
        "false_negatives": len(true_edges - pred_edges),
    }


def evaluate_graph(adj_true: np.ndarray, adj_pred: np.ndarray) -> dict:
    """Full evaluation of a predicted graph against ground truth."""
    shd = structural_hamming_distance(adj_true, adj_pred)
    prf = edge_precision_recall_f1(adj_true, adj_pred)
    return {
        "shd": shd,
        **prf,
    }


if __name__ == "__main__":
    # Test with ground truth vs each candidate
    import json

    with open("data/ground_truth_dag.json") as f:
        gt = json.load(f)
    with open("data/candidate_graphs.json") as f:
        cands = json.load(f)

    adj_true = np.array(gt["adjacency_matrix"]["matrix"])

    print("Evaluation of each candidate against ground truth:")
    print("=" * 60)
    for cand in cands["candidates"]:
        adj_pred = np.array(cand["adjacency_matrix"])
        result = evaluate_graph(adj_true, adj_pred)
        print(f"\n{cand['id']} (confidence: {cand['confidence']}):")
        print(f"  SHD:       {result['shd']}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1:        {result['f1']:.3f}")
        print(f"  TP={result['true_positives']} FP={result['false_positives']} FN={result['false_negatives']}")

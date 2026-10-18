import numpy as np


def search_metrics(result, optimum_val=None, target_val=None):
    rows = result.get("rows", [])

    best_curve = np.asarray(
        [float(r["best_val"]) for r in rows],
        dtype=float
    )

    if not len(best_curve):
        best_curve = np.asarray(
            [float(result["best_val_acc"])],
            dtype=float
        )

    out = {
        "best_val_acc": float(result["best_val_acc"]),
        "test_acc": float(result["test_acc"]),
        "search_auc": float(best_curve.mean()),
        "params": float(result["params"]),
        "flops": float(result["flops"]),
    }

    if optimum_val is None:
        out["final_regret"] = None
    else:
        out["final_regret"] = max(
            0.0,
            float(optimum_val - result["best_val_acc"])
        )

    if target_val is None:
        out["evaluations_to_target"] = None
        out["target_reached"] = None
    else:
        hits = np.flatnonzero(best_curve >= float(target_val))

        if len(hits):
            out["evaluations_to_target"] = int(hits[0] + 1)
            out["target_reached"] = 1
        else:
            out["evaluations_to_target"] = None
            out["target_reached"] = 0

    return out

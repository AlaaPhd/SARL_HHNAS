
import time
import pandas as pd
from copy import deepcopy
from pathlib import Path

from config import Config
from benchmark.nats import NATSBenchTSS
from search.sarl import SARLHHNAS


PROJECT = Path("/content/drive/MyDrive/SARL_HHNAS")

NATS_PATH = (
    "/content/drive/MyDrive/NATS_Bench/"
    "TSS_FAST/NATS-tss-v1_0-3ffb9-simple"
)

OUT = PROJECT / "results/final/llh_behavior"
OUT.mkdir(parents=True, exist_ok=True)

RAW = OUT / "sarl_llh_behavior.csv"

DATASETS = [
    "cifar10",
    "cifar100",
    "ImageNet16-120"
]

SEEDS = range(10)
BUDGET = 500


# ============================================================
# Benchmark
# ============================================================

print("Loading NATS-Bench...")

bench = NATSBenchTSS(
    NATS_PATH,
    hp="200",
    fast_mode=True,
    verbose=False,
    trial_seed=False
)

print("NATS-Bench loaded.")


# ============================================================
# Resume
# ============================================================

def completed_runs():

    if not RAW.exists():
        return set()

    df = pd.read_csv(RAW)

    if df.empty:
        return set()

    # A complete SARL trajectory should reach BUDGET.
    last = (
        df.groupby(["dataset", "seed"])["evaluation"]
          .max()
    )

    return {
        (str(ds), int(seed))
        for (ds, seed), ev in last.items()
        if int(ev) >= BUDGET
    }


done = completed_runs()

print("Completed runs:", len(done), "/ 30")


# ============================================================
# Run SARL
# ============================================================

base_cfg = Config(
    budget=BUDGET,
    beta=0.0,
    params_scale=10.0,
    flops_scale=10.0,
    warmup=64,
    batch_size=64
)


for dataset in DATASETS:

    for seed in SEEDS:

        key = (dataset, seed)

        if key in done:
            print("[SKIP]", dataset, seed)
            continue

        cfg = deepcopy(base_cfg)

        cfg.dataset = dataset
        cfg.seed = seed

        model = SARLHHNAS(
            bench,
            cfg
        )

        start = time.perf_counter()

        result = model.run(
            training=True
        )

        elapsed = time.perf_counter() - start

        records = []

        for row in result["rows"]:

            # INIT has no selected LLH.
            if row.get("llh") == "INIT":
                continue

            records.append({
                "dataset": dataset,
                "seed": seed,
                "budget": BUDGET,
                "evaluation": int(row["evaluation"]),
                "llh": row.get("llh"),
                "best_val": float(row["best_val"]),
                "current_val": float(row["current_val"])
            })

        run_df = pd.DataFrame(records)

        if run_df.empty:
            raise RuntimeError(
                f"No LLH records found for {dataset}, seed={seed}. "
                "Inspect result['rows'] before continuing."
            )

        # Append one completed run.
        run_df.to_csv(
            RAW,
            mode="a",
            header=not RAW.exists(),
            index=False
        )

        done.add(key)

        print(
            f"[{len(done)}/30]",
            dataset,
            f"seed={seed}",
            f"best={result['best_val_acc']:.4f}",
            f"time={elapsed:.2f}s"
        )


print()
print("=" * 70)
print("LLH BEHAVIOR EXPERIMENT COMPLETED")
print("=" * 70)
print(RAW)

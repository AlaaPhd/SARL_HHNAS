
import os
import csv
import time
import pandas as pd
from copy import deepcopy
from pathlib import Path

from config import Config
from benchmark.nats import NATSBenchTSS
from experiments import build_method


# ============================================================
# SETTINGS
# ============================================================

PROJECT = Path("/content/drive/MyDrive/SARL_HHNAS")

NATS_PATH = (
    "/content/drive/MyDrive/NATS_Bench/"
    "TSS_FAST/NATS-tss-v1_0-3ffb9-simple"
)

OUT = PROJECT / "results" / "final" / "convergence"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILE = OUT / "convergence_raw.csv"


DATASETS = [
    "cifar10",
    "cifar100",
    "ImageNet16-120"
]

METHODS = [
    "random",
    "rea",
    "reinforce",
    "random-hh",
    "greedy-hh",
    "cf-hh",
    "sarl"
]

SEEDS = range(10)

BUDGET = 500


# ============================================================
# BENCHMARK
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
print()


# ============================================================
# BASE CONFIG
# Same settings as final experiments
# ============================================================

base_cfg = Config(
    budget=BUDGET,
    params_scale=10.0,
    flops_scale=10.0,
    warmup=64,
    batch_size=64
)


# ============================================================
# RESUME SUPPORT
# ============================================================

def completed_runs():

    if not RAW_FILE.exists():
        return set()

    df = pd.read_csv(RAW_FILE)

    if len(df) == 0:
        return set()

    # A run is complete only when evaluation BUDGET exists.
    last = (
        df.groupby(
            ["dataset", "method", "seed"]
        )["evaluation"]
        .max()
    )

    return {
        (str(ds), str(method), int(seed))
        for (ds, method, seed), ev in last.items()
        if int(ev) >= BUDGET
    }


done = completed_runs()

print("Already completed runs:", len(done))
print()


# ============================================================
# SAVE ONE RUN
# ============================================================

def save_trajectory(
    dataset,
    method,
    seed,
    result
):

    rows = result["rows"]

    records = []

    for r in rows:

        records.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "budget": BUDGET,
            "evaluation": int(r["evaluation"]),
            "best_val": float(r["best_val"])
        })

    df = pd.DataFrame(records)

    write_header = not RAW_FILE.exists()

    df.to_csv(
        RAW_FILE,
        mode="a",
        header=write_header,
        index=False
    )


# ============================================================
# RUN
# ============================================================

total = (
    len(DATASETS)
    * len(METHODS)
    * len(list(SEEDS))
)

counter = len(done)


for dataset in DATASETS:

    print()
    print("=" * 72)
    print("DATASET:", dataset)
    print("=" * 72)

    for method in METHODS:

        for seed in SEEDS:

            key = (
                dataset,
                method,
                seed
            )

            if key in done:

                print(
                    "[SKIP]",
                    dataset,
                    method,
                    "seed",
                    seed
                )

                continue

            cfg = deepcopy(base_cfg)

            cfg.dataset = dataset
            cfg.budget = BUDGET
            cfg.seed = seed
            cfg.beta = 0.0

            obj = build_method(
                method,
                bench,
                cfg
            )

            start = time.perf_counter()

            if method == "sarl":

                result = obj.run(
                    training=True
                )

            else:

                result = obj.run()

            elapsed = (
                time.perf_counter()
                - start
            )

            save_trajectory(
                dataset,
                method,
                seed,
                result
            )

            counter += 1

            print(
                f"[{counter}/{total}]",
                dataset,
                method,
                f"seed={seed}",
                f"best={result['best_val_acc']:.4f}",
                f"time={elapsed:.2f}s"
            )


print()
print("=" * 72)
print("CONVERGENCE EXPERIMENT COMPLETED")
print("=" * 72)

print("Saved:")
print(RAW_FILE)

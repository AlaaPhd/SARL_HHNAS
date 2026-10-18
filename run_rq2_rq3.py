
from pathlib import Path
from copy import deepcopy
import time
import pandas as pd

from config import Config
from benchmark.nats import NATSBench
from search.sarl import SARLHHNAS
from search.baselines import HHBaseline
from evaluation import search_metrics


# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/content/drive/MyDrive/SARL_HHNAS")

NATS_DIR = Path(
    "/content/drive/MyDrive/NATS_Bench/"
    "TSS_FAST/NATS-tss-v1_0-3ffb9-simple"
)

OUT = PROJECT / "results" / "final"

RQ2_RAW = OUT / "raw" / "rq2_state_ablation.csv"
RQ2_SUM = OUT / "summaries" / "rq2_state_ablation_summary.csv"

RQ3_RAW = OUT / "raw" / "rq3_cost_aware.csv"
RQ3_SUM = OUT / "summaries" / "rq3_cost_aware_summary.csv"

RQ2_RAW.parent.mkdir(parents=True, exist_ok=True)
RQ2_SUM.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMMON SETTINGS
# ============================================================

DATASET = "cifar100"
BUDGET = 100
SEEDS = range(10)

BASE_CFG = Config(
    dataset=DATASET,
    budget=BUDGET,
    params_scale=10.0,
    flops_scale=10.0,
    warmup=64,
    batch_size=64
)


# ============================================================
# HELPERS
# ============================================================

def append_row(path, row):
    df = pd.DataFrame([row])

    if path.exists():
        df.to_csv(
            path,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            path,
            index=False
        )


def completed(path, columns):

    if not path.exists():
        return set()

    df = pd.read_csv(
        path,
        keep_default_na=False
    )

    keys = set()

    for _, r in df.iterrows():
        key = []

        for col in columns:

            value = r[col]

            if col in ("seed", "budget"):
                value = int(value)
            elif col == "beta":
                value = float(value)
            else:
                value = str(value).strip()

            key.append(value)

        keys.add(tuple(key))

    return keys


# ============================================================
# BENCHMARK
# ============================================================

print("Loading NATS-Bench...")

bench = NATSBench(
    str(NATS_DIR),
    fast_mode=True
)

print("NATS-Bench loaded.")
print()


# ============================================================
# RQ2
# STATE ABLATION
# ============================================================

print("=" * 72)
print("RQ2: STATE ABLATION")
print("=" * 72)


# State:
#
# 0  p_t
# 1  delta p_t
# 2  delta best
# 3  stagnation
# 4  diversity
# 5  remaining budget
# 6  complexity
# 7:13 LLH utilities

STATE_VARIANTS = {

    "full":
        (),

    "no-stagnation":
        (3,),

    "no-diversity":
        (4,),

    "no-budget":
        (5,),

    "no-complexity":
        (6,),

    "no-llh-utilities":
        (7, 8, 9, 10, 11, 12),

    # Keep only the three performance/progress signals.
    "minimal":
        (3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
}


done = completed(
    RQ2_RAW,
    ["variant", "seed", "budget"]
)


for variant, mask in STATE_VARIANTS.items():

    for seed in SEEDS:

        key = (
            variant,
            seed,
            BUDGET
        )

        if key in done:
            continue

        cfg = deepcopy(BASE_CFG)

        cfg.seed = seed
        cfg.beta = 0.0
        cfg.state_mask = mask
        cfg.shuffle_state = False

        q0 = bench.search_queries
        t0q = bench.test_queries

        start = time.perf_counter()

        result = SARLHHNAS(
            bench,
            cfg
        ).run(
            training=True
        )

        elapsed = (
            time.perf_counter() - start
        )

        metrics = search_metrics(result)

        row = {
            "variant": variant,
            "dataset": DATASET,
            "budget": BUDGET,
            "seed": seed,
            "best_val_acc":
                metrics["best_val_acc"],
            "test_acc":
                metrics["test_acc"],
            "search_auc":
                metrics["search_auc"],
            "params":
                metrics["params"],
            "flops":
                metrics["flops"],
            "search_queries":
                bench.search_queries - q0,
            "test_queries":
                bench.test_queries - t0q,
            "time_sec":
                elapsed
        }

        append_row(
            RQ2_RAW,
            row
        )

        done.add(key)

        print(
            "RQ2",
            variant,
            "seed", seed,
            "| val",
            f'{metrics["best_val_acc"]:.4f}',
            "| AUC",
            f'{metrics["search_auc"]:.4f}'
        )


# ------------------------------------------------------------
# Negative control: shuffled state
# ------------------------------------------------------------

variant = "state-negative-control"

for seed in SEEDS:

    key = (
        variant,
        seed,
        BUDGET
    )

    if key in done:
        continue

    cfg = deepcopy(BASE_CFG)

    cfg.seed = seed
    cfg.beta = 0.0
    cfg.state_mask = ()
    cfg.shuffle_state = True

    q0 = bench.search_queries
    t0q = bench.test_queries

    start = time.perf_counter()

    result = SARLHHNAS(
        bench,
        cfg
    ).run(
        training=True
    )

    elapsed = (
        time.perf_counter() - start
    )

    metrics = search_metrics(result)

    row = {
        "variant":
            variant,
        "dataset":
            DATASET,
        "budget":
            BUDGET,
        "seed":
            seed,
        "best_val_acc":
            metrics["best_val_acc"],
        "test_acc":
            metrics["test_acc"],
        "search_auc":
            metrics["search_auc"],
        "params":
            metrics["params"],
        "flops":
            metrics["flops"],
        "search_queries":
            bench.search_queries - q0,
        "test_queries":
            bench.test_queries - t0q,
        "time_sec":
            elapsed
    }

    append_row(
        RQ2_RAW,
        row
    )

    done.add(key)

    print(
        "RQ2",
        variant,
        "seed", seed,
        "| val",
        f'{metrics["best_val_acc"]:.4f}',
        "| AUC",
        f'{metrics["search_auc"]:.4f}'
    )


# ============================================================
# RQ2 SUMMARY
# ============================================================

rq2 = pd.read_csv(RQ2_RAW)

rq2_summary = (
    rq2
    .groupby("variant")
    .agg(
        n_runs=("seed", "count"),

        best_val_mean=(
            "best_val_acc",
            "mean"
        ),
        best_val_std=(
            "best_val_acc",
            "std"
        ),

        test_mean=(
            "test_acc",
            "mean"
        ),
        test_std=(
            "test_acc",
            "std"
        ),

        search_auc_mean=(
            "search_auc",
            "mean"
        ),
        search_auc_std=(
            "search_auc",
            "std"
        ),

        time_mean=(
            "time_sec",
            "mean"
        )
    )
    .reset_index()
)

rq2_summary.to_csv(
    RQ2_SUM,
    index=False
)

print()
print("RQ2 SUMMARY")
print(rq2_summary.to_string(index=False))


# ============================================================
# RQ3
# COST-AWARE SEARCH
# ============================================================

print()
print("=" * 72)
print("RQ3: COST-AWARE SEARCH")
print("=" * 72)


# Accuracy-complexity trade-off strengths.
#
# beta = 0.00 is already represented by RQ1.
# Positive beta values explicitly penalize complexity.

BETAS = [
    0.01,
    0.05,
    0.10
]

METHODS = [
    "random-hh",
    "greedy-hh",
    "cf-hh",
    "sarl"
]


done3 = completed(
    RQ3_RAW,
    [
        "method",
        "beta",
        "seed",
        "budget"
    ]
)


for beta in BETAS:

    for method in METHODS:

        for seed in SEEDS:

            key = (
                method,
                float(beta),
                seed,
                BUDGET
            )

            if key in done3:
                continue

            cfg = deepcopy(BASE_CFG)

            cfg.seed = seed
            cfg.beta = beta
            cfg.state_mask = ()
            cfg.shuffle_state = False

            if method == "random-hh":

                obj = HHBaseline(
                    bench,
                    cfg,
                    mode="random"
                )

            elif method == "greedy-hh":

                obj = HHBaseline(
                    bench,
                    cfg,
                    mode="greedy"
                )

            elif method == "cf-hh":

                obj = HHBaseline(
                    bench,
                    cfg,
                    mode="choice"
                )

            elif method == "sarl":

                obj = SARLHHNAS(
                    bench,
                    cfg
                )

            else:
                raise ValueError(method)

            q0 = bench.search_queries
            t0q = bench.test_queries

            start = time.perf_counter()

            if method == "sarl":
                result = obj.run(
                    training=True
                )
            else:
                result = obj.run()

            elapsed = (
                time.perf_counter() - start
            )

            metrics = search_metrics(result)

            row = {
                "method":
                    method,
                "dataset":
                    DATASET,
                "budget":
                    BUDGET,
                "beta":
                    beta,
                "seed":
                    seed,

                "best_val_acc":
                    metrics["best_val_acc"],

                "test_acc":
                    metrics["test_acc"],

                "search_auc":
                    metrics["search_auc"],

                "params":
                    metrics["params"],

                "flops":
                    metrics["flops"],

                "search_queries":
                    bench.search_queries - q0,

                "test_queries":
                    bench.test_queries - t0q,

                "time_sec":
                    elapsed
            }

            append_row(
                RQ3_RAW,
                row
            )

            done3.add(key)

            print(
                "RQ3",
                "beta", beta,
                method,
                "seed", seed,
                "| val",
                f'{metrics["best_val_acc"]:.4f}',
                "| params",
                f'{metrics["params"]:.4f}',
                "| FLOPs",
                f'{metrics["flops"]:.4f}'
            )


# ============================================================
# RQ3 SUMMARY
# ============================================================

rq3 = pd.read_csv(RQ3_RAW)

rq3_summary = (
    rq3
    .groupby(
        [
            "beta",
            "method"
        ]
    )
    .agg(
        n_runs=(
            "seed",
            "count"
        ),

        best_val_mean=(
            "best_val_acc",
            "mean"
        ),
        best_val_std=(
            "best_val_acc",
            "std"
        ),

        test_mean=(
            "test_acc",
            "mean"
        ),
        test_std=(
            "test_acc",
            "std"
        ),

        params_mean=(
            "params",
            "mean"
        ),
        params_std=(
            "params",
            "std"
        ),

        flops_mean=(
            "flops",
            "mean"
        ),
        flops_std=(
            "flops",
            "std"
        ),

        search_auc_mean=(
            "search_auc",
            "mean"
        ),
        search_auc_std=(
            "search_auc",
            "std"
        ),

        time_mean=(
            "time_sec",
            "mean"
        )
    )
    .reset_index()
)

rq3_summary.to_csv(
    RQ3_SUM,
    index=False
)

print()
print("RQ3 SUMMARY")
print(rq3_summary.to_string(index=False))


print()
print("=" * 72)
print("RQ2 + RQ3 COMPLETED")
print("=" * 72)

print("RQ2 raw:")
print(RQ2_RAW)

print("RQ2 summary:")
print(RQ2_SUM)

print("RQ3 raw:")
print(RQ3_RAW)

print("RQ3 summary:")
print(RQ3_SUM)

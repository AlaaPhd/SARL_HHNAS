
import os
import sys
import csv
import copy
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config
from benchmark.nats import NATSBenchTSS
from experiments import build_method
from training import meta_train
from search.sarl import SARLHHNAS
from evaluation import search_metrics


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    "/content/drive/MyDrive/SARL_HHNAS"
)

NATS_DIR = Path(
    "/content/drive/MyDrive/NATS_Bench/"
    "TSS_FAST/NATS-tss-v1_0-3ffb9-simple"
)

ROOT = PROJECT / "results" / "final"

RAW = ROOT / "raw"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
SUMMARIES = ROOT / "summaries"
FIGURES = ROOT / "figures"

for p in [
    ROOT,
    RAW,
    CHECKPOINTS,
    LOGS,
    SUMMARIES,
    FIGURES
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FINAL EXPERIMENT SETTINGS
# ============================================================

DATASETS = [
    "cifar10",
    "cifar100",
    "ImageNet16-120"
]

BUDGETS = [
    100,
    250,
    500
]

# 10 independent search seeds.
# Increase to 30 later if required by final protocol.
SEARCH_SEEDS = list(range(10))

# Independent policy seeds.
POLICY_SEEDS = [
    1000,
    1001,
    1002
]

BASELINES = [
    "random",
    "rea",
    "reinforce",
    "random-hh",
    "greedy-hh",
    "cf-hh"
]

META_EPISODES = 100

META_EPISODE_BUDGET = 100

SOURCE_DATASET = "cifar10"


# ============================================================
# CONFIG
# ============================================================

base_cfg = Config(
    params_scale=10.0,
    flops_scale=10.0,
    warmup=64,
    batch_size=64
)


# ============================================================
# BENCHMARK
# ============================================================

bench = NATSBenchTSS(
    path=str(NATS_DIR),
    hp="200",
    fast_mode=True,
    verbose=False,
    trial_seed=False
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SAFE APPEND
# ============================================================

def append_csv(path, row):

    path = Path(path)

    exists = path.exists()

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# COMPLETED RUN DETECTION
# ============================================================

def completed_keys(path):

    path = Path(path)

    if not path.exists():
        return set()

    df = pd.read_csv(
        path,
        keep_default_na=False
    )

    keys = set()

    for _, r in df.iterrows():

        dataset = str(
            r["dataset"]
        ).strip()

        budget = int(
            r["budget"]
        )

        search_seed = int(
            r["search_seed"]
        )

        method = str(
            r["method"]
        ).strip()

        source_dataset = str(
            r.get("source_dataset", "")
        ).strip()

        policy_seed = str(
            r.get("policy_seed", "")
        ).strip()

        if policy_seed:
            try:
                policy_seed = str(
                    int(float(policy_seed))
                )
            except ValueError:
                pass

        keys.add(
            (
                dataset,
                budget,
                search_seed,
                method,
                source_dataset,
                policy_seed
            )
        )

    return keys

# ============================================================
# POLICY CHECKPOINT
# ============================================================

def save_agent(agent, path, metadata):

    payload = {
        "online":
            agent.online.state_dict(),

        "target":
            agent.target.state_dict(),

        "optimizer":
            agent.opt.state_dict(),

        "steps":
            agent.steps,

        "metadata":
            metadata
    }

    torch.save(
        payload,
        path
    )


def load_agent(agent, path):

    payload = torch.load(
        path,
        map_location=agent.device
    )

    agent.online.load_state_dict(
        payload["online"]
    )

    agent.target.load_state_dict(
        payload["target"]
    )

    agent.opt.load_state_dict(
        payload["optimizer"]
    )

    agent.steps = int(
        payload["steps"]
    )

    return agent


# ============================================================
# META TRAIN POLICIES
# ============================================================

def prepare_meta_policies():

    print("\n" + "=" * 78)
    print("META-TRAINING")
    print("=" * 78)

    paths = []

    for pseed in POLICY_SEEDS:

        checkpoint = (
            CHECKPOINTS /
            f"meta_{SOURCE_DATASET}_seed{pseed}.pt"
        )

        paths.append(
            (
                pseed,
                checkpoint
            )
        )

        if checkpoint.exists():

            print(
                "Checkpoint exists:",
                checkpoint.name
            )

            continue

        cfg = copy.deepcopy(
            base_cfg
        )

        cfg.dataset = SOURCE_DATASET
        cfg.seed = pseed
        cfg.beta = 0.0

        seed_everything(
            pseed
        )

        q0 = bench.search_queries
        t0 = bench.test_queries

        start = time.time()

        agent, summaries = meta_train(
            bench,
            cfg,
            episodes=META_EPISODES,
            episode_budget=META_EPISODE_BUDGET
        )

        elapsed = (
            time.time()
            -
            start
        )

        meta_queries = (
            bench.search_queries
            -
            q0
        )

        meta_test_queries = (
            bench.test_queries
            -
            t0
        )

        if meta_test_queries != 0:
            raise RuntimeError(
                "TEST LEAKAGE DURING META-TRAINING"
            )

        metadata = {
            "source_dataset":
                SOURCE_DATASET,

            "policy_seed":
                pseed,

            "episodes":
                META_EPISODES,

            "episode_budget":
                META_EPISODE_BUDGET,

            "meta_queries":
                meta_queries,

            "meta_time_sec":
                elapsed,

            "mean_episode_best":
                float(
                    np.mean(
                        summaries
                    )
                ),

            "last_episode_best":
                float(
                    summaries[-1]
                )
        }

        save_agent(
            agent,
            checkpoint,
            metadata
        )

        pd.DataFrame({
            "episode":
                np.arange(
                    1,
                    len(summaries) + 1
                ),

            "best_val":
                summaries
        }).to_csv(
            LOGS /
            f"meta_curve_seed{pseed}.csv",
            index=False
        )

        print(
            f"Saved policy {pseed} | "
            f"queries={meta_queries} | "
            f"time={elapsed:.2f}s"
        )

    return paths


# ============================================================
# RUN FINAL EXPERIMENTS
# ============================================================

def run_main_experiments(
    policy_paths
):

    output = (
        RAW /
        "main_results.csv"
    )

    done = completed_keys(
        output
    )

    # --------------------------------------------------------
    # DIRECT + HH BASELINES + SARL ONLINE
    # --------------------------------------------------------

    methods = (
        BASELINES
        +
        ["sarl-online"]
    )

    for dataset in DATASETS:

        for budget in BUDGETS:

            for search_seed in SEARCH_SEEDS:

                for method in methods:

                    key = (
                        dataset,
                        budget,
                        search_seed,
                        method,
                        "",
                        ""
                    )

                    if key in done:
                        continue

                    seed_everything(
                        search_seed
                    )

                    cfg = copy.deepcopy(
                        base_cfg
                    )

                    cfg.dataset = dataset
                    cfg.budget = budget
                    cfg.seed = search_seed
                    cfg.beta = 0.0

                    q0 = bench.search_queries
                    tq0 = bench.test_queries

                    start = time.time()

                    if method == "sarl-online":

                        obj = build_method(
                            "sarl",
                            bench,
                            cfg
                        )

                        result = obj.run(
                            training=True,
                            query_test=True
                        )

                    else:

                        obj = build_method(
                            method,
                            bench,
                            cfg
                        )

                        result = obj.run()

                    elapsed = (
                        time.time()
                        -
                        start
                    )

                    metrics = search_metrics(
                        result
                    )

                    row = {
                        "dataset":
                            dataset,

                        "budget":
                            budget,

                        "search_seed":
                            search_seed,

                        "method":
                            method,

                        "source_dataset":
                            "",

                        "policy_seed":
                            "",

                        "meta_queries":
                            0,

                        "search_queries":
                            bench.search_queries
                            -
                            q0,

                        "test_queries":
                            bench.test_queries
                            -
                            tq0,

                        "search_time_sec":
                            elapsed,

                        **metrics
                    }

                    append_csv(
                        output,
                        row
                    )

                    done.add(key)

                    print(
                        dataset,
                        budget,
                        search_seed,
                        method,
                        "val=",
                        round(
                            metrics[
                                "best_val_acc"
                            ],
                            4
                        )
                    )

    # --------------------------------------------------------
    # META POLICY:
    # FROZEN + FINE-TUNED
    # --------------------------------------------------------

    from agents.dqn import DDQNAgent

    for pseed, checkpoint in policy_paths:

        for dataset in DATASETS:

            for budget in BUDGETS:

                for search_seed in SEARCH_SEEDS:

                    for mode in [
                        "sarl-meta-frozen",
                        "sarl-meta-finetune"
                    ]:

                        key = (
                            dataset,
                            budget,
                            search_seed,
                            mode,
                            SOURCE_DATASET,
                            str(pseed)
                        )

                        if key in done:
                            continue

                        seed_everything(
                            search_seed
                        )

                        cfg = copy.deepcopy(
                            base_cfg
                        )

                        cfg.dataset = dataset
                        cfg.budget = budget
                        cfg.seed = search_seed
                        cfg.beta = 0.0

                        agent = DDQNAgent(
                            cfg
                        )

                        agent = load_agent(
                            agent,
                            checkpoint
                        )

                        q0 = bench.search_queries
                        tq0 = bench.test_queries

                        start = time.time()

                        search = SARLHHNAS(
                            bench,
                            cfg,
                            agent=agent
                        )

                        if mode == "sarl-meta-frozen":

                            result = search.run(
                                training=False,
                                query_test=True
                            )

                        else:

                            result = search.run(
                                training=True,
                                query_test=True
                            )

                        elapsed = (
                            time.time()
                            -
                            start
                        )

                        metrics = search_metrics(
                            result
                        )

                        row = {
                            "dataset":
                                dataset,

                            "budget":
                                budget,

                            "search_seed":
                                search_seed,

                            "method":
                                mode,

                            "source_dataset":
                                SOURCE_DATASET,

                            "policy_seed":
                                pseed,

                            "meta_queries":
                                (
                                    META_EPISODES
                                    *
                                    META_EPISODE_BUDGET
                                ),

                            "search_queries":
                                bench.search_queries
                                -
                                q0,

                            "test_queries":
                                bench.test_queries
                                -
                                tq0,

                            "search_time_sec":
                                elapsed,

                            **metrics
                        }

                        append_csv(
                            output,
                            row
                        )

                        done.add(key)

                        print(
                            dataset,
                            budget,
                            search_seed,
                            mode,
                            "policy=",
                            pseed,
                            "val=",
                            round(
                                metrics[
                                    "best_val_acc"
                                ],
                                4
                            )
                        )

    return output


# ============================================================
# SUMMARIES
# ============================================================

def make_summaries(
    result_file
):

    df = pd.read_csv(
        result_file
    )

    summary = (
        df.groupby(
            [
                "dataset",
                "budget",
                "method"
            ],
            as_index=False
        )
        .agg(
            n_runs=(
                "best_val_acc",
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

            search_auc_mean=(
                "search_auc",
                "mean"
            ),

            search_auc_std=(
                "search_auc",
                "std"
            ),

            params_mean=(
                "params",
                "mean"
            ),

            flops_mean=(
                "flops",
                "mean"
            ),

            time_mean=(
                "search_time_sec",
                "mean"
            )
        )
    )

    summary.to_csv(
        SUMMARIES /
        "main_summary.csv",
        index=False
    )

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 78)
    print("SARL-HHNAS FINAL EXPERIMENT RUNNER")
    print("=" * 78)

    print(
        "Architectures:",
        len(bench.api)
    )

    print(
        "Datasets:",
        DATASETS
    )

    print(
        "Budgets:",
        BUDGETS
    )

    print(
        "Search seeds:",
        SEARCH_SEEDS
    )

    print(
        "Policy seeds:",
        POLICY_SEEDS
    )

    policies = (
        prepare_meta_policies()
    )

    results = (
        run_main_experiments(
            policies
        )
    )

    summary = (
        make_summaries(
            results
        )
    )

    print("\n" + "=" * 78)
    print("COMPLETED")
    print("=" * 78)

    print(
        "Results:",
        results
    )

    print(
        "Summary:",
        SUMMARIES /
        "main_summary.csv"
    )

    print(
        "Search queries:",
        bench.search_queries
    )

    print(
        "Test queries:",
        bench.test_queries
    )

    print("\nSummary:")
    print(summary.to_string(index=False))

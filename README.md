# SARL-HHNAS

## State-Aware Reinforcement Learning Hyper-Heuristic for Cost-Efficient Neural Architecture Search in Deep Classification

**SARL-HHNAS** is a research framework for Neural Architecture Search (NAS) that combines **reinforcement learning** with **hyper-heuristic search**.

Instead of directly using reinforcement learning to construct a neural architecture, SARL-HHNAS learns **which low-level search heuristic should be applied according to the current state of the architecture search process**.

The main idea is:

> **Learn search knowledge, not only an architecture.**

The framework uses a **Double Deep Q-Network (DDQN)** as a high-level heuristic selector. At every search step, the DDQN observes the current search state and selects one of six low-level architecture search heuristics. The selected heuristic modifies the current neural architecture, and the resulting architecture is evaluated using **NATS-Bench**.

The implementation supports accuracy-oriented search, cost-aware search, state-ablation experiments, multiple search baselines, and transfer of learned heuristic-selection policies between classification tasks.

---

## Overview

Traditional NAS methods generally search directly over neural architectures. This can require a large number of architecture evaluations and may repeatedly apply similar search operations without explicitly learning which search strategy is suitable for the current search condition.

SARL-HHNAS introduces an additional decision level:

```text
Search State
     │
     ▼
┌───────────────────────┐
│   DDQN High-Level     │
│   Heuristic Selector  │
└───────────┬───────────┘
            │
            ▼
      Select H1 ... H6
            │
            ▼
┌───────────────────────┐
│ Low-Level NAS         │
│ Heuristic             │
└───────────┬───────────┘
            │
            ▼
     Candidate Architecture
            │
            ▼
       NATS-Bench
            │
            ▼
 Validation Performance
            │
            ▼
 Objective / Reward
            │
            ▼
 Acceptance Decision
            │
            └──────────► Search State
```

The high-level reinforcement-learning agent therefore learns a **heuristic-selection policy** rather than directly predicting an architecture.

---

# Main Features

The framework currently provides:

- State-aware hyper-heuristic neural architecture search
- Double Deep Q-Network (DDQN) heuristic selection
- Six low-level NAS heuristics
- NATS-Bench topology search space support
- NAS-Bench-201 architecture representation
- CIFAR-10 experiments
- CIFAR-100 experiments
- ImageNet16-120 experiments
- Validation-only architecture selection
- Final test evaluation after search
- Duplicate architecture avoidance
- Probabilistic acceptance of non-improving solutions
- Accuracy-oriented search
- Cost-aware architecture search
- Policy meta-training
- Frozen policy transfer
- Fine-tuned policy transfer
- State-ablation experiments
- Multiple NAS and hyper-heuristic baselines
- Query-budget-controlled experiments
- Search trajectory logging
- Reproducible random seeds
- Experiment checkpointing and resumption

---

# Search Space

SARL-HHNAS uses the **NATS-Bench Topology Search Space (TSS)**, corresponding to the NAS-Bench-201 topology space.

The search space contains:

```text
15,625 architectures
```

Each architecture is represented as a four-node cell containing six directed edges.

Each edge can contain one of five operations:

```text
none
skip_connect
nor_conv_1x1
nor_conv_3x3
avg_pool_3x3
```

An architecture is internally represented as:

```python
Architecture(
    ops=(
        op_1,
        op_2,
        op_3,
        op_4,
        op_5,
        op_6
    )
)
```

The six operations correspond to the NAS-Bench-201 edge order:

```text
0 → 1
0 → 2
1 → 2
0 → 3
1 → 3
2 → 3
```

---

# Benchmark Tasks

Experiments are supported on three classification tasks:

| Dataset | Search dataset used by NATS-Bench |
|---|---|
| CIFAR-10 | `cifar10-valid` |
| CIFAR-100 | `cifar100` |
| ImageNet16-120 | `ImageNet16-120` |

During architecture search, SARL-HHNAS uses **validation performance only**.

Test accuracy is not used for:

- reinforcement-learning rewards;
- architecture acceptance;
- incumbent selection;
- heuristic selection;
- meta-training;
- policy learning.

The test performance of the selected architecture is queried only after the architecture search has finished.

This separation prevents test-set information from influencing the search procedure.

---

# SARL-HHNAS Architecture

The framework contains two main search levels.

## High-Level Search

A DDQN agent determines which low-level heuristic should be applied.

The neural network used by the agent is:

```text
13-dimensional search state
        │
        ▼
Linear(13, 128)
        │
       ReLU
        │
Linear(128, 128)
        │
       ReLU
        │
Linear(128, 6)
        │
        ▼
Q(H1), Q(H2), ..., Q(H6)
```

The action with the highest estimated Q-value is selected during exploitation.

During training, epsilon-greedy exploration is used.

---

# Search State

The high-level agent observes a 13-dimensional state vector:

```text
[p_t,
 Δp_t,
 Δ_best,
 g_t,
 d_t,
 b_t,
 c_t,
 u1,
 u2,
 u3,
 u4,
 u5,
 u6]
```

The state describes several aspects of the current search process, including:

- current architecture performance;
- recent performance change;
- difference from the current best solution;
- search stagnation;
- architecture diversity;
- remaining search budget;
- architecture complexity;
- historical utility of each low-level heuristic.

This allows the agent to select heuristics according to the current search condition instead of using a fixed heuristic-selection strategy.

---

# Low-Level Heuristics

Six low-level heuristics are available.

## H1 — Single-Edge Mutation

Randomly selects one architecture edge and replaces its operation with another valid operation.

```text
H1_single
```

## H2 — Two-Edge Mutation

Randomly selects two different edges and mutates both operations.

```text
H2_double
```

## H3 — Operation Swap

Selects two edges containing different operations and exchanges their operations.

```text
H3_swap
```

## H4 — Best-Guided Mutation

Selects an edge that differs from the current best architecture and replaces its operation with the corresponding operation from the best architecture.

```text
H4_guided
```

## H5 — Three-Edge Perturbation

Randomly selects three architecture edges and mutates their operations.

```text
H5_triple
```

## H6 — Diversity Restart

Generates a random architecture sufficiently different from the current best architecture.

```text
H6_restart
```

These heuristics provide different levels of exploration and exploitation.

---

# Duplicate Handling

The framework attempts to avoid wasting benchmark evaluations on previously evaluated architectures.

When a low-level heuristic generates a duplicate architecture, the framework resamples the heuristic output before querying NATS-Bench.

The current procedure uses:

```text
20 low-level heuristic attempts
```

followed by up to:

```text
200 random unvisited architecture attempts
```

if necessary.

Duplicate generation itself does not consume the architecture evaluation budget.

---

# Search Objective

The general architecture objective is:

\[
J(A)=P_{\mathrm{val}}(A)-\beta C(A)
\]

where:

- \(P_{\mathrm{val}}(A)\) is validation performance;
- \(C(A)\) is normalized architecture complexity;
- \(\beta\) controls the trade-off between predictive performance and computational cost.

When:

\[
\beta=0,
\]

the search focuses on validation performance.

When:

\[
\beta>0,
\]

architecture complexity is incorporated into the search objective.

The complexity term can incorporate architecture parameters and FLOPs.

---

# Reward

After applying a selected heuristic, the reinforcement-learning agent receives a reward based on the change in architecture objective.

Conceptually:

\[
R_t =
\operatorname{clip}
\left(
\kappa
[J(A_{t+1})-J(A_t)]
+
R_{\mathrm{best}},
-1,
1
\right)
\]

where \(R_{\mathrm{best}}\) provides an additional signal when a new best architecture is discovered.

Reward clipping limits large updates and stabilizes DDQN learning.

---

# Move Acceptance

SARL-HHNAS separates **heuristic selection** from **move acceptance**.

The reinforcement-learning agent selects the low-level heuristic.

The acceptance mechanism then determines whether the generated architecture becomes the current architecture.

An improving architecture is accepted directly.

A non-improving architecture may still be accepted probabilistically according to a temperature-dependent rule.

This mechanism allows the search to escape local search regions while preserving exploitation of promising architectures.

---

# Double Deep Q-Network

The current DDQN configuration includes:

| Parameter | Value |
|---|---:|
| State dimension | 13 |
| Number of actions | 6 |
| Hidden units | 128 |
| Hidden layers | 2 |
| Discount factor \(\gamma\) | 0.95 |
| Learning rate | \(10^{-3}\) |
| Replay-buffer size | 10,000 |
| Batch size | 64 |
| Initial epsilon | 1.00 |
| Final epsilon | 0.05 |
| Epsilon decay steps | 5,000 |
| Target-network update | 100 |
| Gradient clipping | 1.0 |
| Warm-up size | 64 |

DDQN uses the online network for action selection and the target network for target-value estimation.

---

# Policy Meta-Training

SARL-HHNAS supports learning heuristic-selection knowledge before applying the policy to a target search task.

For example:

```text
CIFAR-10
    │
    ▼
Meta-training
    │
    ▼
Learned DDQN heuristic-selection policy
    │
    ├────────► CIFAR-100
    │
    └────────► ImageNet16-120
```

Importantly, source-task meta-training uses validation information only.

No final test query is performed during policy meta-training.

---

# Transfer Modes

Three SARL configurations can be studied.

## SARL-Online

The DDQN policy learns during the target architecture search without using a previously meta-trained policy.

```text
SARL-Online
```

## SARL-Meta-Frozen

A policy learned on a source task is transferred to the target task without updating its learned parameters during target search.

```text
SARL-Meta-Frozen
```

This evaluates direct transfer of heuristic-selection knowledge.

## SARL-Meta-Finetune

A meta-trained policy initializes the target search and continues learning during the target task.

```text
SARL-Meta-Finetune
```

This evaluates whether transferred heuristic-selection knowledge can be adapted to the target search environment.

---

# Search Cost

The framework distinguishes the cost of target architecture search from the cost of learning the reusable search policy.

Online search cost is defined as:

\[
C_{\mathrm{online}}=C_{\mathrm{search}}.
\]

The total cost including meta-training is:

\[
C_{\mathrm{total}}
=
C_{\mathrm{meta}}
+
C_{\mathrm{search}}.
\]

When a learned policy is reused across \(n\) target searches, its amortized cost can be expressed as:

\[
C_{\mathrm{amortized}}(n)
=
\frac{C_{\mathrm{meta}}}{n}
+
C_{\mathrm{search}}.
\]

This distinction prevents offline policy-learning cost from being hidden when comparing search methods.

---

# Baselines

The framework currently supports the following search methods:

```text
Random Search
Regularized Evolution (REA)
REINFORCE
Random-HH
Greedy-HH
CF-HH
SARL-Online
SARL-Meta-Frozen
SARL-Meta-Finetune
```

### Random-HH

Randomly selects one of the six low-level heuristics.

### Greedy-HH

Uses historical heuristic rewards and selects the heuristic with the highest estimated mean reward.

### CF-HH

Uses a choice-function-style score combining:

- individual heuristic performance;
- pairwise heuristic information;
- heuristic recency.

`CF-HH` in this repository should be interpreted as the implemented choice-function-style hyper-heuristic baseline. It is not presented as an exact reproduction of another published algorithm unless explicitly stated.

---

# Experimental Questions

The framework is designed to investigate the following research questions.

### RQ1 — Search Effectiveness

How effectively does SARL-HHNAS search for high-performing neural architectures compared with direct NAS and hyper-heuristic baselines?

### RQ2 — State Awareness

How much does the state-aware representation contribute to the performance of reinforcement-learning heuristic selection?

### RQ3 — Cost-Aware Search

How does incorporating architecture complexity into the objective affect the trade-off between predictive performance and computational cost?

### RQ4 — Search Efficiency

How efficiently does SARL-HHNAS use a limited architecture-evaluation budget compared with alternative search strategies?

### RQ5 — Policy Transfer

To what extent can heuristic-selection knowledge learned on one neural architecture search task generalize or adapt to unseen classification tasks?

---

# Evaluation Metrics

The experimental framework supports metrics including:

- Best validation accuracy
- Final selected-architecture test accuracy
- Search AUC
- Validation regret
- Evaluations-to-target
- Search time
- Controller time
- Number of architecture queries
- Parameters
- FLOPs
- Architecture complexity
- Heuristic-selection frequency
- Heuristic utility
- Search trajectory

### Search AUC

Search AUC summarizes the quality of the incumbent architecture throughout the search budget rather than considering only the final architecture.

This is useful for evaluating how quickly a method discovers strong architectures.

---

# Experimental Reproducibility

The implementation separates:

```text
policy seed
```

from:

```text
search seed
```

to distinguish variation caused by policy learning from variation caused by target architecture search.

The DDQN implementation also uses seeded random-number generation for:

- epsilon-greedy action selection;
- replay-buffer sampling;
- NumPy operations;
- PyTorch initialization and learning.

Experiment results are saved incrementally so interrupted experiments can be resumed without rerunning completed configurations.

---

# Project Structure

```text
SARL_HHNAS/
│
├── agents/
│   └── dqn.py
│
├── benchmark/
│   ├── architecture.py
│   └── nats.py
│
├── heuristics/
│   └── llhs.py
│
├── search/
│   ├── core.py
│   ├── sarl.py
│   ├── baselines.py
│   └── direct_baselines.py
│
├── config.py
├── evaluation.py
├── experiments.py
├── training.py
├── research_suite.py
├── run_research.py
├── run_final_experiments.py
│
└── results/
    └── final/
        ├── checkpoints/
        ├── logs/
        ├── raw/
        ├── summaries/
        ├── statistics/
        └── figures/
```

---

# Installation

The implementation is written in Python and uses PyTorch and the NATS-Bench API.

A typical environment requires:

```bash
pip install nats-bench torch numpy pandas matplotlib scipy
```

Clone the repository:

```bash
git clone <repository-url>
cd SARL_HHNAS
```

The NATS-Bench data must be obtained separately from the official NATS-Bench distribution.

---

# NATS-Bench Setup

For memory-efficient experiments, the project can use the extracted NATS-Bench TSS simple archive with:

```python
from nats_bench import create

api = create(
    "/path/to/NATS-tss-v1_0-3ffb9-simple",
    "tss",
    fast_mode=True,
    verbose=False
)
```

The expected topology search space contains:

```text
15625 architectures
```

Using the simple archive with `fast_mode=True` avoids loading the complete benchmark database into memory.

---

# Running the Experiments

After configuring the benchmark path, the main experiment runner can be executed with:

```bash
python run_final_experiments.py
```

The runner supports incremental result saving and can skip completed configurations after interruption.

Typical experiment settings include:

```text
Datasets:
    CIFAR-10
    CIFAR-100
    ImageNet16-120

Search budgets:
    100
    250
    500

Methods:
    Random
    REA
    REINFORCE
    Random-HH
    Greedy-HH
    CF-HH
    SARL-Online
    SARL-Meta-Frozen
    SARL-Meta-Finetune
```

Experimental settings used for a particular publication should be taken from the corresponding experiment configuration and released result files rather than assumed from this README.

---

# Output

Experiment outputs are stored under:

```text
results/final/
```

The directory can contain:

```text
checkpoints/
    Meta-trained DDQN policies

logs/
    Meta-training and search trajectories

raw/
    Individual experimental runs

summaries/
    Aggregated experimental results

statistics/
    Statistical-test outputs

figures/
    Publication figures
```

---

# Example Meta-Training

A policy can be meta-trained on a source classification task before transfer.

Conceptually:

```python
agent, summaries = meta_train(
    benchmark,
    config,
    episodes=100,
    episode_budget=100
)
```

The meta-training stage queries validation performance only.

A learned policy can subsequently be evaluated using frozen or fine-tuned transfer.

---

# Reproducibility and Fair Comparison

To support fair comparison:

1. Search methods operate under predefined architecture-query budgets.
2. Duplicate architectures are handled before benchmark evaluation where applicable.
3. Search decisions use validation performance rather than test performance.
4. Test accuracy is queried only for the final selected architecture.
5. Independent random seeds are used for repeated experiments.
6. Policy-training and target-search randomness are distinguished.
7. Meta-training cost is reported separately from online search cost.
8. Raw experimental results are retained for statistical analysis.

---

# Current Development Status

This repository contains the research implementation of SARL-HHNAS.

The framework is under active experimental development. Therefore, numerical results in temporary notebooks, smoke tests, or pilot runs should not be interpreted as final reported results.

Final experimental results should be taken from the released final result files and the associated research paper.

---

# Citation

If you use this repository in academic work, please cite the associated paper after its bibliographic information becomes available.

```bibtex
@article{SARLHHNAS,
  title   = {State-Aware Reinforcement Learning Hyper-Heuristic for
             Cost-Efficient Neural Architecture Search in Deep Classification},
  author  = {Saba Alaa Abdulwahhab and co-authors},
  journal = {To appear},
  year    = {2026}
}
```

The citation entry should be updated after publication with the final journal, volume, pages/article number, and DOI.

---

# Research Scope

SARL-HHNAS is intended for research on:

- Neural Architecture Search
- Automated Machine Learning
- Reinforcement Learning
- Hyper-Heuristics
- Search Control
- Deep Classification
- Cost-Aware NAS
- Transferable Search Policies

The implementation is designed primarily for controlled benchmark experiments and research reproducibility.

---

# License

Please refer to the repository's `LICENSE` file for the terms governing use, modification, and redistribution of this source code.

NATS-Bench and any external datasets or software dependencies remain subject to their respective licenses and terms of use.

---

# Contact

For questions concerning the research implementation, experiments, or academic use of SARL-HHNAS, please use the repository's GitHub Issues section or contact the corresponding authors of the associated paper.

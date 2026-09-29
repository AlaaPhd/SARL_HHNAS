# SARL_HHNAS
# SARL-HHNAS

**State-Aware Reinforcement Learning Hyper-Heuristic for Neural Architecture Search**

SARL-HHNAS is a research framework for adaptive and cost-efficient Neural Architecture Search (NAS) for deep classification. The framework combines reinforcement learning with a selection hyper-heuristic to dynamically choose low-level search heuristics according to the current state of the architecture search process.

Instead of using reinforcement learning to directly generate complete neural architectures, SARL-HHNAS uses a reinforcement learning controller to select among a set of low-level heuristics. These heuristics modify candidate architectures through different local and exploratory search operations.

The framework is designed for experiments using the **NATS-Bench Topology Search Space (TSS)** and supports classification tasks including CIFAR-10, CIFAR-100, and ImageNet16-120.

## Main Features

- State-aware reinforcement learning for adaptive heuristic selection.
- Double Deep Q-Network (Double DQN) controller.
- Six complementary low-level heuristics for architecture modification.
- Validation-based architecture search with test data isolated from the search process.
- Performance-only and cost-aware search objectives.
- Architecture complexity based on model parameters and FLOPs.
- Adaptive move acceptance for balancing exploration and exploitation.
- Multiple search budgets for evaluating search efficiency.
- Policy training and transfer across classification tasks.
- State and heuristic-selection ablation experiments.
- Search trajectory and heuristic-selection logging.
- Support for statistical comparison of NAS methods.
- Integration with NATS-Bench TSS.

## Framework Overview

The main search process is:

```text
Classification Dataset
        |
        v
NATS-Bench Search Space
        |
        v
Current Architecture
        |
        v
Search-State Extraction
        |
        v
Double DQN Controller
        |
        v
Low-Level Heuristic Selection
        |
        v
Candidate Architecture
        |
        v
Validation Evaluation
        |
        v
Reward and Move Acceptance
        |
        v
Policy Update
```

The controller learns **which search heuristic to apply under different search conditions**, rather than directly predicting the final neural architecture.

## Search State

The state representation contains information about the current search condition, including:

- Current validation performance.
- Recent performance change.
- Difference from the best architecture found.
- Search stagnation.
- Recent architecture diversity.
- Remaining search budget.
- Architecture complexity.
- Recent utility of each low-level heuristic.

These features allow the controller to adapt heuristic selection as the search progresses.

## Low-Level Heuristics

SARL-HHNAS currently uses six architecture-search heuristics:

1. Single-edge mutation.
2. Two-edge mutation.
3. Operation swap.
4. Best-guided mutation.
5. Three-edge perturbation.
6. Diversity-based restart.

The same heuristic pool can be used by different hyper-heuristic selectors to support controlled comparisons.

## Supported Search Methods

The experimental framework includes:

- Random Search.
- Regularized Evolution (REA).
- REINFORCE-based NAS.
- Random Hyper-Heuristic.
- Greedy Hyper-Heuristic.
- Choice-Function-style Hyper-Heuristic.
- SARL-HHNAS.

## Benchmark

The current implementation targets the **NATS-Bench Topology Search Space (TSS)**.

Supported classification datasets include:

```text
CIFAR-10
CIFAR-100
ImageNet16-120
```

The TSS contains 15,625 candidate neural architectures constructed from five operations over a four-node cell.

The NATS-Bench benchmark file is **not included in this repository** because of its size. It should be downloaded separately from the official NATS-Bench repository.

Expected benchmark file:

```text
NATS-tss-v1_0-3ffb9.pickle.pbz2
```

## Test-Data Isolation

During architecture search, SARL-HHNAS uses validation performance for:

- reward calculation,
- architecture comparison,
- heuristic selection,
- move acceptance, and
- incumbent selection.

Test accuracy is intended only for the final selected architecture after completion of the search. This separation is used to reduce test-data leakage during NAS experiments.

## Project Structure

```text
SARL_HHNAS/
│
├── agents/
│   └── Reinforcement-learning components
│
├── benchmark/
│   └── NATS-Bench interface and architecture representation
│
├── heuristics/
│   └── Low-level architecture-search heuristics
│
├── search/
│   └── SARL-HHNAS and baseline search algorithms
│
├── tests/
│   └── Unit and experimental validation tests
│
├── config.py
├── evaluation.py
├── experiments.py
├── research_suite.py
├── training.py
├── analysis_results.py
├── plot_results.py
├── run_natsbench.py
├── run_research.py
├── main.py
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SARL_HHNAS.git
cd SARL_HHNAS
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
pytest -q
```

## Experimental Goals

The framework is designed to investigate the following questions:

- Can state-aware reinforcement learning improve heuristic selection during neural architecture search?
- Which search-state features are most useful for adaptive heuristic selection?
- Can SARL-HHNAS balance classification performance with architecture complexity and search cost?
- How does the framework behave under different architecture-evaluation budgets?
- Can learned heuristic-selection knowledge generalize or adapt across classification tasks?

## Research Status

**Current status: Experimental research implementation.**

The framework and testing infrastructure are under active development. Experimental results should not be considered final until the complete NATS-Bench experiments, statistical analysis, ablation studies, and policy-transfer experiments have been completed.

## Citation

A citation will be added after publication of the associated research paper.

## License

Please add the appropriate software license before public distribution.

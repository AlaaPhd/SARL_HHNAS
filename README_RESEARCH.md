# SARL-HHNAS Research Runner

## Install
`pip install -r requirements.txt`

## NATS-Bench
Use the official TSS benchmark file/archive. The search adapter never exposes test accuracy.
During search:
- CIFAR-10 uses `cifar10-valid` / `x-valid`.
- CIFAR-100 uses `cifar100` / `x-valid`.
- ImageNet16-120 uses `ImageNet16-120` / `x-valid`.

After the budget is exhausted, the selected incumbent is evaluated once:
- CIFAR-10: `cifar10` / `ori-test`
- CIFAR-100: `cifar100` / `x-test`
- ImageNet16-120: `ImageNet16-120` / `x-test`

## Smoke test
`pytest -q`

## Real RQ1
`python run_natsbench.py --benchmark /path/to/NATS-tss-v1_0-3ffb9.pickle.pbz2 --rq 1 --seeds 30`

## All experiments
`python run_natsbench.py --benchmark /path/to/NATS-tss-v1_0-3ffb9.pickle.pbz2 --rq all --seeds 30 --meta-episodes 500`

MockBenchmark values are software-test data only and must never be reported as research results.


## Budget fairness
SARL-HHNAS and HH baselines resample duplicate candidates before benchmark
evaluation. A generated duplicate does not consume an architecture-query budget.
The final test query is outside the search budget and is performed once after
validation-based architecture selection.

## Transfer seed hierarchy
RQ5 separates:
- `policy_seed`: independent source-task DDQN meta-training.
- `search_seed`: independent target-task search trajectory.

Recommended initial design: 5 policy seeds x 10 target search seeds for frozen
and fine-tuned transfer. Scratch is evaluated once per target search seed.

Example:
`python run_natsbench.py --benchmark /path/to/NATS-tss-v1_0-3ffb9.pickle.pbz2 --rq 5 --policy-seeds 5 --transfer-search-seeds 10 --meta-episodes 500`

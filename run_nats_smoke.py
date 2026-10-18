
from config import Config
from benchmark.nats import NATSBenchTSS
from research_suite import rq1

NATS_DIR = "/content/drive/MyDrive/NATS_Bench/TSS_FAST/NATS-tss-v1_0-3ffb9-simple"

print("=" * 70)
print("SARL-HHNAS - REAL NATS-Bench Smoke Test")
print("=" * 70)

# Load real NATS-Bench
bench = NATSBenchTSS(
    path=NATS_DIR,
    hp="200",
    fast_mode=True,
    verbose=False,
    trial_seed=False
)

print("NATS-Bench loaded successfully")
print("Number of architectures:", len(bench.api))
print("Initial search queries:", bench.search_queries)
print("Initial test queries:", bench.test_queries)

# Configuration
cfg = Config(
    params_scale=10.0,
    flops_scale=10.0,
    warmup=8,
    batch_size=8
)

# One real experiment only:
# Dataset = CIFAR-10
# Budget  = 100
# Seed    = 0
# Method  = SARL-HHNAS
rq1(
    bench,
    cfg,
    ["cifar10"],
    [100],
    [0],
    ["sarl"],
    "results/rq1_nats_smoke.csv"
)

print("=" * 70)
print("Smoke test completed")
print("Search queries:", bench.search_queries)
print("Test queries:", bench.test_queries)
print("=" * 70)

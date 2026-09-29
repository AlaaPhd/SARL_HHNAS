
from config import Config
from benchmark.mock import MockBenchmark
from research_suite import rq1, rq2_state_ablation, rq3_cost

if __name__=="__main__":
    # Replace MockBenchmark with NATSBenchBenchmark for real experiments.
    bench=MockBenchmark()
    cfg=Config(params_scale=10.0, flops_scale=10.0, warmup=8, batch_size=8)

    rq1(bench,cfg,["cifar10"],[25],[0,1],
        ["random","rea","reinforce","random-hh","greedy-hh","cf-hh","sarl"],
        "results/rq1_smoke.csv")

    rq2_state_ablation(bench,cfg,"cifar10",25,[0],
        "results/rq2_smoke.csv")

    rq3_cost(bench,cfg,"cifar10",25,[0],[0.0,0.01,0.05],
        "results/rq3_smoke.csv")

    print("Smoke experiments completed. These mock results are for software testing only.")

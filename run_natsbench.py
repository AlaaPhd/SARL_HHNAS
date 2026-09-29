
import argparse, os
from config import Config
from benchmark.nats import NATSBenchTSS
from research_suite import rq1, rq2_state_ablation, rq3_cost, rq4_budget, rq5_transfer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark",required=True,help="NATS-tss-v1_0-3ffb9.pickle.pbz2 or simple archive")
    ap.add_argument("--out",default="results")
    ap.add_argument("--rq",choices=["1","2","3","4","5","all"],default="1")
    ap.add_argument("--hp",default="200",choices=["12","200"])
    ap.add_argument("--seeds",type=int,default=30)
    ap.add_argument("--meta-episodes",type=int,default=500)
    ap.add_argument("--policy-seeds",type=int,default=5)
    ap.add_argument("--transfer-search-seeds",type=int,default=10)
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    bench=NATSBenchTSS(args.benchmark,hp=args.hp,trial_seed=False)
    cfg=Config()
    seeds=list(range(args.seeds))
    methods=["random","rea","reinforce","random-hh","greedy-hh","cf-hh","sarl"]
    datasets=["cifar10","cifar100","ImageNet16-120"]
    budgets=[100,250,500]

    if args.rq in ("1","all"):
        rq1(bench,cfg,datasets,budgets,seeds,methods,os.path.join(args.out,"rq1.csv"))
    if args.rq in ("2","all"):
        for ds in datasets:
            rq2_state_ablation(bench,cfg,ds,250,seeds,os.path.join(args.out,f"rq2_{ds}.csv"))
    if args.rq in ("3","all"):
        for ds in datasets:
            rq3_cost(bench,cfg,ds,250,seeds,[0.0,0.01,0.05,0.10],
                     os.path.join(args.out,f"rq3_{ds}.csv"))
    if args.rq in ("4","all"):
        for ds in datasets:
            rq4_budget(bench,cfg,ds,budgets,seeds,methods,os.path.join(args.out,f"rq4_{ds}.csv"))
    if args.rq in ("5","all"):
        for target in ["cifar100","ImageNet16-120"]:
            rq5_transfer(bench,cfg,"cifar10",target,list(range(args.policy_seeds)),list(range(args.transfer_search_seeds)),args.meta_episodes,100,250,
                         os.path.join(args.out,f"rq5_cifar10_to_{target}.csv"))
    print("Finished. Search queries:",bench.search_queries,"post-search test queries:",bench.test_queries)

if __name__=="__main__": main()

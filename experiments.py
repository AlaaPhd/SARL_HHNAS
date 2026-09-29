import csv, os
from copy import deepcopy
from search.sarl import SARLHHNAS
from search.baselines import HHBaseline
from search.direct_baselines import RandomSearch, RegularizedEvolution, ReinforceNAS
from evaluation import search_metrics

METHODS=('random','rea','reinforce','random-hh','greedy-hh','cf-hh','sarl')

def build_method(name, bench, cfg):
    if name=='random': return RandomSearch(bench,cfg)
    if name=='rea': return RegularizedEvolution(bench,cfg)
    if name=='reinforce': return ReinforceNAS(bench,cfg)
    if name in ('random-hh','greedy-hh','cf-hh'):
        return HHBaseline(bench,cfg,{'random-hh':'random','greedy-hh':'greedy','cf-hh':'choice'}[name])
    if name=='sarl': return SARLHHNAS(bench,cfg)
    raise ValueError(name)

def run_grid(bench, base_cfg, datasets=('cifar10','cifar100','ImageNet16-120'), budgets=(100,250,500), seeds=range(30), methods=METHODS, output='results/summary.csv'):
    records=[]
    for ds in datasets:
      for budget in budgets:
       for seed in seeds:
        for method in methods:
            cfg=deepcopy(base_cfg); cfg.dataset=ds; cfg.budget=int(budget); cfg.seed=int(seed)
            obj=build_method(method,bench,cfg)
            result=obj.run() if method!='sarl' else obj.run(training=True)
            rec={'method':method,'dataset':ds,'budget':budget,'seed':seed,**search_metrics(result)}
            records.append(rec)
    if records:
        os.makedirs(os.path.dirname(output) or '.',exist_ok=True)
        with open(output,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    return records

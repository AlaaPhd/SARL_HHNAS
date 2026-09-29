import argparse
from config import Config
from benchmark.mock import MockBenchmark
from benchmark.nats import NATSBenchTSS
from search.sarl import SARLHHNAS
from search.baselines import HHBaseline
from search.direct_baselines import RandomSearch,RegularizedEvolution,ReinforceNAS
from evaluation import search_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=['mock','nats'],default='mock'); p.add_argument('--path')
    p.add_argument('--dataset',default='cifar10'); p.add_argument('--budget',type=int,default=100)
    p.add_argument('--method',choices=['sarl','random-hh','greedy-hh','cf-hh','random','rea','reinforce'],default='sarl')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--log',default='results/run.csv')
    a=p.parse_args(); cfg=Config(dataset=a.dataset,budget=a.budget,seed=a.seed)
    bench=MockBenchmark() if a.benchmark=='mock' else NATSBenchTSS(a.path)
    if a.method=='sarl': result=SARLHHNAS(bench,cfg).run(training=True,log_path=a.log)
    elif a.method in ('random-hh','greedy-hh','cf-hh'):
        result=HHBaseline(bench,cfg,{'random-hh':'random','greedy-hh':'greedy','cf-hh':'choice'}[a.method]).run()
    else:
        cls={'random':RandomSearch,'rea':RegularizedEvolution,'reinforce':ReinforceNAS}[a.method]; result=cls(bench,cfg).run()
    print(search_metrics(result))
    print('best_arch:',result['best_arch'].to_nats_string())
if __name__=='__main__': main()

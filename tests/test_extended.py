from config import Config
from benchmark.mock import MockBenchmark
from search.direct_baselines import RandomSearch,RegularizedEvolution,ReinforceNAS
from training import meta_train,frozen_transfer
from evaluation import search_metrics

def test_direct_baselines_and_metrics():
    b=MockBenchmark(); c=Config(budget=20,seed=3)
    for cls in (RandomSearch,RegularizedEvolution,ReinforceNAS):
        r=cls(b,c).run(); m=search_metrics(r)
        assert len(r['rows'])==20
        assert 0 < m['best_val_acc'] <= 100

def test_meta_train_and_frozen():
    b=MockBenchmark(); c=Config(budget=10,seed=2,batch_size=4,warmup=4,hidden=16)
    agent,h=meta_train(b,c,episodes=2,episode_budget=10)
    assert len(h)==2
    r=frozen_transfer(b,c,agent)
    assert r['best_val_acc']>0

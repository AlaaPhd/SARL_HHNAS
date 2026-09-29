
from config import Config
from benchmark.mock import MockBenchmark
from research_suite import rq2_state_ablation, rq3_cost

def test_rq2_small(tmp_path):
    b=MockBenchmark()
    c=Config(budget=8,seed=1,warmup=2,batch_size=2)
    rows=rq2_state_ablation(b,c,"cifar10",8,[1],str(tmp_path/"rq2.csv"))
    assert len(rows)==8
    assert (tmp_path/"rq2.csv").exists()

def test_rq3_small(tmp_path):
    b=MockBenchmark()
    c=Config(budget=8,seed=1,warmup=2,batch_size=2,params_scale=10,flops_scale=10)
    rows=rq3_cost(b,c,"cifar10",8,[1],[0,.05],str(tmp_path/"rq3.csv"))
    assert len(rows)==2

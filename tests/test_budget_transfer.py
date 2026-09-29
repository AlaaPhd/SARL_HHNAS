
from config import Config
from benchmark.mock import MockBenchmark
from search.sarl import SARLHHNAS
from research_suite import rq5_transfer

class CountBench(MockBenchmark):
    def __init__(self): self.search=0; self.test=0
    def evaluate(self,a,dataset="cifar10"):
        self.search+=1; return super().evaluate(a,dataset)
    def final_test(self,a,dataset="cifar10"):
        self.test+=1; return super().final_test(a,dataset)

def test_unique_query_budget():
    b=CountBench(); c=Config(budget=30,seed=3,warmup=2,batch_size=2)
    r=SARLHHNAS(b,c).run(training=True)
    assert b.search == 30
    assert len(r["rows"]) == 30
    assert b.test == 1

def test_nested_transfer_counts(tmp_path):
    b=MockBenchmark(); c=Config(budget=6,seed=0,warmup=2,batch_size=2)
    rows=rq5_transfer(b,c,"cifar10","cifar100",[0,1],[10,11],2,5,6,str(tmp_path/"t.csv"))
    # 2 policy seeds * 2 search seeds * (frozen + finetune) + 2 scratch
    assert len(rows)==10
    assert sum(r["mode"]=="scratch" for r in rows)==2

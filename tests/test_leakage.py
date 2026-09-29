
from config import Config
from benchmark.mock import MockBenchmark
from search.sarl import SARLHHNAS

class AuditBenchmark(MockBenchmark):
    def __init__(self): self.search=0; self.tests=0
    def evaluate(self,a,dataset="cifar10"):
        self.search+=1
        x=super().evaluate(a,dataset)
        assert "test_acc" not in x
        return x
    def final_test(self,a,dataset="cifar10"):
        self.tests+=1
        return super().final_test(a,dataset)

def test_test_is_post_search_only():
    b=AuditBenchmark(); c=Config(budget=12,seed=2,warmup=2,batch_size=2)
    r=SARLHHNAS(b,c).run(training=True)
    assert b.tests==1
    assert b.search<=c.budget
    assert "test_acc" in r

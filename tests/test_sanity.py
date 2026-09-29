import random
from benchmark.architecture import Architecture
from heuristics.llhs import LLHS
from benchmark.mock import MockBenchmark
from config import Config
from search.sarl import SARLHHNAS

def test_llhs():
    r=random.Random(1); a=Architecture.random(r); best=Architecture.random(r)
    for h in LLHS:
        c=h(a,best,r); assert len(c.ops)==6; assert c!=a or h.__name__=='h3'

def test_short_search():
    c=Config(budget=20,seed=1,warmup=8,batch_size=8)
    x=SARLHHNAS(MockBenchmark(),c).run(training=True)
    assert len(x['rows'])==20
    assert 0<=x['test_acc']<=100

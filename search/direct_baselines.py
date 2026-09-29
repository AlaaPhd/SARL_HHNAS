import random
from collections import deque
import numpy as np
import torch
from torch import nn
from benchmark.architecture import Architecture, OPS


def _result(benchmark, cfg, best, bm, rows):
    test = benchmark.final_test(best,cfg.dataset) if hasattr(benchmark,'final_test') else bm.get('test_acc', float('nan'))
    return {'best_arch':best,'best_val_acc':bm['val_acc'],'test_acc':float(test),
            'params':bm.get('params',0.0),'flops':bm.get('flops',0.0),'rows':rows}

class RandomSearch:
    def __init__(self, benchmark, cfg):
        self.b=benchmark; self.c=cfg; self.rng=random.Random(cfg.seed)
    def run(self):
        best=None; bm=None; rows=[]
        for t in range(1,self.c.budget+1):
            a=Architecture.random(self.rng); m=self.b.evaluate(a,self.c.dataset)
            if bm is None or m['val_acc']>bm['val_acc']: best,bm=a,m
            rows.append({'evaluation':t,'best_val':bm['val_acc'],'current_val':m['val_acc']})
        return _result(self.b,self.c,best,bm,rows)

class RegularizedEvolution:
    def __init__(self, benchmark, cfg, population_size=25, tournament_size=5):
        self.b=benchmark; self.c=cfg; self.rng=random.Random(cfg.seed)
        self.population_size=min(population_size,max(2,cfg.budget//4)); self.tournament_size=tournament_size
    def run(self):
        pop=deque(); rows=[]; best=None; bm=None; t=0
        while t<min(self.population_size,self.c.budget):
            a=Architecture.random(self.rng); m=self.b.evaluate(a,self.c.dataset); t+=1; pop.append((a,m))
            if bm is None or m['val_acc']>bm['val_acc']: best,bm=a,m
            rows.append({'evaluation':t,'best_val':bm['val_acc'],'current_val':m['val_acc']})
        while t<self.c.budget:
            sample=self.rng.sample(list(pop),min(self.tournament_size,len(pop)))
            parent=max(sample,key=lambda x:x[1]['val_acc'])[0]
            child=parent.mutate([self.rng.randrange(6)],self.rng); m=self.b.evaluate(child,self.c.dataset); t+=1
            pop.append((child,m)); pop.popleft()  # aging/regularization
            if m['val_acc']>bm['val_acc']: best,bm=child,m
            rows.append({'evaluation':t,'best_val':bm['val_acc'],'current_val':m['val_acc']})
        return _result(self.b,self.c,best,bm,rows)

class ReinforceNAS:
    '''Factorized REINFORCE controller over the 6 TSS edges. One sampled architecture = one query.'''
    def __init__(self, benchmark, cfg, lr=0.05, entropy_weight=1e-3):
        self.b=benchmark; self.c=cfg; torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.logits=nn.Parameter(torch.zeros(6,len(OPS))); self.opt=torch.optim.Adam([self.logits],lr=lr)
        self.entropy_weight=entropy_weight; self.baseline=None
    def run(self, training=True):
        rows=[]; best=None; bm=None
        for t in range(1,self.c.budget+1):
            dist=torch.distributions.Categorical(logits=self.logits)
            idx=dist.sample(); a=Architecture(tuple(OPS[i] for i in idx.tolist())); m=self.b.evaluate(a,self.c.dataset)
            reward=m['val_acc']/100.0
            if training:
                self.baseline=reward if self.baseline is None else 0.9*self.baseline+0.1*reward
                advantage=reward-self.baseline
                loss=-(dist.log_prob(idx).sum()*advantage + self.entropy_weight*dist.entropy().sum())
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            if bm is None or m['val_acc']>bm['val_acc']: best,bm=a,m
            rows.append({'evaluation':t,'best_val':bm['val_acc'],'current_val':m['val_acc']})
        return _result(self.b,self.c,best,bm,rows)

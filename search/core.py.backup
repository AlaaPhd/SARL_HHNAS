import math, random
import numpy as np
from collections import deque
from benchmark.architecture import Architecture
from heuristics.llhs import LLHS,LLH_NAMES

class SearchCore:
    def __init__(self,benchmark,cfg):
        self.b=benchmark; self.c=cfg; self.rng=random.Random(cfg.seed)
        self.utilities=np.zeros(6,dtype=np.float32); self.history=deque(maxlen=cfg.history_window)
        self.cache={}; self.params_min=self.params_max=None; self.flops_min=self.flops_max=None
    def eval(self,a):
        k=(a.ops,self.c.dataset)
        if k not in self.cache: self.cache[k]=self.b.evaluate(a,self.c.dataset)
        return self.cache[k]
    def unique_candidate(self, action, current, best, max_attempts=20):
        # Duplicate generation is free; only a genuinely queried architecture consumes budget.
        # If an LLH repeatedly generates visited candidates, fall back to an unvisited random TSS point.
        for _ in range(max_attempts):
            cand=LLHS[action](current,best,self.rng)
            if (cand.ops,self.c.dataset) not in self.cache:
                return cand
        for _ in range(200):
            cand=Architecture.random(self.rng)
            if (cand.ops,self.c.dataset) not in self.cache:
                return cand
        return LLHS[action](current,best,self.rng)

    def complexity(self,m):
        # Reference-scale normalization. Set scales from benchmark metadata/pilot data
        # before RQ3; beta=0 in RQ1, so complexity does not affect core search.
        p=max(0.0,float(m.get('params',0.0)))
        f=max(0.0,float(m.get('flops',0.0)))
        ps=max(1e-12,float(self.c.params_scale))
        fs=max(1e-12,float(self.c.flops_scale))
        return 0.5*min(1.0,p/ps)+0.5*min(1.0,f/fs)
    def objective(self,m): return m['val_acc']/100.0-self.c.beta*self.complexity(m)
    def diversity(self,a):
        return 1.0 if not self.history else float(np.mean([a.distance(x) for x in self.history]))
    def state(self,current,cm,best,bm,prev_perf,stagnation,evals):
        p=cm['val_acc']/100.; dp=np.clip(p-prev_perf,-1,1); db=np.clip(p-bm['val_acc']/100.,-1,1)
        g=min(1.,stagnation/max(1,self.c.stagnation_max)); d=self.diversity(current); rem=max(0.,1-evals/self.c.budget)
        comp=self.complexity(cm)
        s=np.asarray([p,dp,db,g,d,rem,comp,*np.clip(self.utilities,-1,1)],dtype=np.float32)
        # RQ2 feature ablation: indices in state_mask are zeroed without changing DQN shape.
        for idx in getattr(self.c,'state_mask',()):
            if 0 <= int(idx) < len(s): s[int(idx)]=0.0
        # State-shuffle control: deterministic permutation per search object.
        if getattr(self.c,'shuffle_state',False):
            # Negative control: destroy stable feature semantics across observations.
            perm=list(range(len(s))); self.rng.shuffle(perm)
            s=s[np.asarray(perm,dtype=int)]
        return s
    def temperature(self,evals):
        x=evals/max(1,self.c.budget); return self.c.t0*((self.c.tmin/self.c.t0)**x)

    def finalize(self,best,bm,rows):
        test = self.b.final_test(best,self.c.dataset) if hasattr(self.b,'final_test') else bm.get('test_acc', float('nan'))
        return {'best_arch':best,'best_val_acc':bm['val_acc'],'test_acc':float(test),
                'params':bm.get('params',0.0),'flops':bm.get('flops',0.0),'rows':rows}

    def accept(self,old,new,evals):
        if new>=old: return True
        # objective is near [0,1]; scale deterioration to percentage-point-like units
        loss=100*(old-new); prob=math.exp(-loss/max(1e-9,self.temperature(evals)))
        return self.rng.random()<prob

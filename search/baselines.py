import numpy as np
from benchmark.architecture import Architecture
from heuristics.llhs import LLHS,LLH_NAMES
from search.core import SearchCore

class HHBaseline(SearchCore):
    def __init__(self,b,c,mode='random'):
        super().__init__(b,c); self.mode=mode; self.count=np.zeros(6); self.sum_reward=np.zeros(6); self.last=np.zeros(6); self.prev_action=None; self.pair=np.zeros((6,6))
    def choose(self,t):
        if self.mode=='random': return self.rng.randrange(6)
        untried=np.where(self.count==0)[0]
        if len(untried): return int(untried[0])
        means=self.sum_reward/np.maximum(1,self.count)
        if self.mode=='greedy': return int(np.argmax(means))
        # Simple documented Choice-Function-style implementation: individual + pairwise + recency.
        recency=(t-self.last)/max(1,t)
        pair=np.zeros(6) if self.prev_action is None else self.pair[self.prev_action]
        score=means+0.5*pair+0.5*recency
        return int(np.argmax(score))
    def run(self):
        cur=Architecture.random(self.rng); cm=self.eval(cur); best=cur; bm=cm; self.history.append(cur); evals=1; stagn=0; rows=[{'evaluation':1,'best_val':bm['val_acc'],'current_val':cm['val_acc'],'llh':'INIT'}]
        while evals<self.c.budget:
            a=self.choose(evals); cand=self.unique_candidate(a,cur,best); nm=self.eval(cand); evals+=1
            oj,nj,bj=self.objective(cm),self.objective(nm),self.objective(bm); r=float(np.clip((nj-oj)*100,-1,1)); accepted=self.accept(oj,nj,evals)
            if accepted: cur,cm=cand,nm
            if nj>bj: best,bm=cand,nm; stagn=0
            else: stagn+=1
            self.count[a]+=1; self.sum_reward[a]+=r
            if self.prev_action is not None: self.pair[self.prev_action,a]=0.7*self.pair[self.prev_action,a]+0.3*r
            self.last[a]=evals; self.prev_action=a; self.history.append(cur)
            rows.append({'evaluation':evals,'best_val':bm['val_acc'],'current_val':cm['val_acc'],'llh':LLH_NAMES[a]})
        return self.finalize(best,bm,rows)

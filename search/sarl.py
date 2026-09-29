import csv, os
import numpy as np
from benchmark.architecture import Architecture
from heuristics.llhs import LLHS,LLH_NAMES
from agents.dqn import DDQNAgent
from search.core import SearchCore

class SARLHHNAS(SearchCore):
    def __init__(self,benchmark,cfg,agent=None):
        super().__init__(benchmark,cfg); self.agent=agent or DDQNAgent(cfg)
    def run(self,training=True,log_path=None):
        cur=Architecture.random(self.rng); cm=self.eval(cur); best=cur; bm=cm; evals=1; stagn=0; prev=cm['val_acc']/100.; rows=[{'evaluation':1,'arch':cur.to_nats_string(),'llh':'INIT','reward':0.0,'accepted':1,'current_val':cm['val_acc'],'best_val':bm['val_acc'],'epsilon':'','loss':''}]
        self.history.append(cur)
        while evals<self.c.budget:
            s=self.state(cur,cm,best,bm,prev,stagn,evals); action,eps=self.agent.act(s,training)
            cand=self.unique_candidate(action,cur,best); nm=self.eval(cand); evals+=1
            oldj=self.objective(cm); newj=self.objective(nm); bestj=self.objective(bm)
            reward=(newj-oldj)*100.0 + (self.c.discovery_bonus if newj>bestj else 0.0)
            reward=float(np.clip(reward,-1,1)); accepted=self.accept(oldj,newj,evals)
            old_perf=cm['val_acc']/100.
            if accepted: cur,cm=cand,nm
            if newj>bestj: best,bm=cand,nm; stagn=0
            else: stagn+=1
            self.utilities[action]=(1-self.c.utility_alpha)*self.utilities[action]+self.c.utility_alpha*reward
            self.history.append(cur)
            ns=self.state(cur,cm,best,bm,old_perf,stagn,evals); done=evals>=self.c.budget
            loss=None
            if training:
                self.agent.buffer.push(s,action,reward,ns,done); loss=self.agent.learn()
            rows.append({'evaluation':evals,'arch':cand.to_nats_string(),'llh':LLH_NAMES[action],'reward':reward,'accepted':int(accepted),'current_val':cm['val_acc'],'best_val':bm['val_acc'],'epsilon':eps,'loss':loss if loss is not None else ''})
            prev=old_perf
        if log_path:
            os.makedirs(os.path.dirname(log_path) or '.',exist_ok=True)
            with open(log_path,'w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        return self.finalize(best,bm,rows)


from copy import deepcopy
import csv, os, itertools
from experiments import build_method
from evaluation import search_metrics
from training import meta_train, frozen_transfer, finetuned_transfer

STATE_ABLATIONS = {
    "full": (),
    "no_stagnation": (3,),
    "no_diversity": (4,),
    "no_budget": (5,),
    "no_complexity": (6,),
    "no_llh_history": (7,8,9,10,11,12),
    "minimal": (2,3,4,5,6,7,8,9,10,11,12),
}

def _write(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def rq1(bench,cfg,datasets,budgets,seeds,methods,output):
    rows=[]
    for ds,budget,seed,method in itertools.product(datasets,budgets,seeds,methods):
        c=deepcopy(cfg); c.dataset=ds; c.budget=budget; c.seed=seed; c.beta=0.0
        obj=build_method(method,bench,c)
        res=obj.run() if method!="sarl" else obj.run(training=True)
        rows.append({"rq":"RQ1","dataset":ds,"budget":budget,"seed":seed,
                     "method":method,**search_metrics(res)})
    _write(rows,output); return rows

def rq2_state_ablation(bench,cfg,dataset,budget,seeds,output):
    rows=[]
    for name,mask in STATE_ABLATIONS.items():
        for seed in seeds:
            c=deepcopy(cfg); c.dataset=dataset; c.budget=budget; c.seed=seed
            c.beta=0.0; c.state_mask=mask
            res=build_method("sarl",bench,c).run(training=True)
            rows.append({"rq":"RQ2","variant":name,"dataset":dataset,"budget":budget,
                         "seed":seed,**search_metrics(res)})
    # negative control
    for seed in seeds:
        c=deepcopy(cfg); c.dataset=dataset; c.budget=budget; c.seed=seed
        c.beta=0.0; c.shuffle_state=True
        res=build_method("sarl",bench,c).run(training=True)
        rows.append({"rq":"RQ2","variant":"shuffled_state","dataset":dataset,"budget":budget,
                     "seed":seed,**search_metrics(res)})
    _write(rows,output); return rows

def rq3_cost(bench,cfg,dataset,budget,seeds,betas,output):
    rows=[]
    for beta,seed in itertools.product(betas,seeds):
        c=deepcopy(cfg); c.dataset=dataset; c.budget=budget; c.seed=seed; c.beta=float(beta)
        res=build_method("sarl",bench,c).run(training=True)
        rows.append({"rq":"RQ3","beta":beta,"dataset":dataset,"budget":budget,
                     "seed":seed,**search_metrics(res)})
    _write(rows,output); return rows

def rq4_budget(bench,cfg,dataset,budgets,seeds,methods,output):
    return rq1(bench,cfg,[dataset],budgets,seeds,methods,output)

def rq5_transfer(bench,cfg,source,target,policy_seeds,search_seeds,
                 episodes,episode_budget,target_budget,output):
    """Nested transfer design: independently trained policies x independent target searches."""
    rows=[]
    for pseed in policy_seeds:
        src=deepcopy(cfg); src.dataset=source; src.seed=pseed
        source_agent,source_summary=meta_train(
            bench,src,episodes=episodes,episode_budget=episode_budget
        )
        # Freeze a checkpoint state so frozen and fine-tuned branches start identically.
        import copy
        frozen_agent=copy.deepcopy(source_agent)
        finetune_agent_template=copy.deepcopy(source_agent)

        for sseed in search_seeds:
            target_cfg=deepcopy(cfg); target_cfg.dataset=target
            target_cfg.seed=sseed; target_cfg.budget=target_budget

            r=frozen_transfer(bench,target_cfg,copy.deepcopy(frozen_agent))
            rows.append({"rq":"RQ5","source":source,"target":target,"mode":"frozen",
                         "policy_seed":pseed,"search_seed":sseed,
                         "meta_episodes":episodes,"meta_queries":episodes*episode_budget,
                         **search_metrics(r)})

            r=finetuned_transfer(bench,target_cfg,copy.deepcopy(finetune_agent_template))
            rows.append({"rq":"RQ5","source":source,"target":target,"mode":"finetune",
                         "policy_seed":pseed,"search_seed":sseed,
                         "meta_episodes":episodes,"meta_queries":episodes*episode_budget,
                         **search_metrics(r)})

    # Scratch is independent of policy seed and is run once per target search seed.
    for sseed in search_seeds:
        target_cfg=deepcopy(cfg); target_cfg.dataset=target
        target_cfg.seed=sseed; target_cfg.budget=target_budget
        r=build_method("sarl",bench,target_cfg).run(training=True)
        rows.append({"rq":"RQ5","source":source,"target":target,"mode":"scratch",
                     "policy_seed":None,"search_seed":sseed,
                     "meta_episodes":0,"meta_queries":0,**search_metrics(r)})
    _write(rows,output); return rows

from copy import deepcopy
from config import Config
from agents.dqn import DDQNAgent
from search.sarl import SARLHHNAS

def meta_train(benchmark, cfg, episodes=500, episode_budget=100):
    train_cfg=deepcopy(cfg); train_cfg.budget=episode_budget
    agent=DDQNAgent(train_cfg)
    summaries=[]
    for ep in range(episodes):
        train_cfg.seed=cfg.seed+ep
        result=SARLHHNAS(
            benchmark,
            train_cfg,
            agent=agent
        ).run(
            training=True,
            query_test=False
        )
        summaries.append(result['best_val_acc'])
    return agent, summaries

def frozen_transfer(benchmark, cfg, agent):
    return SARLHHNAS(benchmark,cfg,agent=agent).run(training=False)

def finetuned_transfer(benchmark, cfg, agent):
    return SARLHHNAS(benchmark,cfg,agent=agent).run(training=True)

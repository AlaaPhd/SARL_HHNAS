import math, random
from collections import deque, namedtuple
import numpy as np
import torch
from torch import nn

Transition=namedtuple('Transition','state action reward next_state done')
class ReplayBuffer:
    def __init__(self,n): self.data=deque(maxlen=n)
    def push(self,*x): self.data.append(Transition(*x))
    def __len__(self): return len(self.data)
    def sample(self,n): return random.sample(self.data,n)

class QNet(nn.Module):
    def __init__(self,state_dim=13,actions=6,hidden=128):
        super().__init__(); self.net=nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,actions))
    def forward(self,x): return self.net(x)

class DDQNAgent:
    def __init__(self,cfg,device=None):
        self.cfg=cfg; self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.online=QNet(hidden=cfg.hidden).to(self.device); self.target=QNet(hidden=cfg.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt=torch.optim.Adam(self.online.parameters(),lr=cfg.lr)
        self.buffer=ReplayBuffer(cfg.replay_size); self.steps=0
    def epsilon(self):
        return self.cfg.epsilon_end+(self.cfg.epsilon_start-self.cfg.epsilon_end)*math.exp(-self.steps/max(1,self.cfg.epsilon_decay_steps))
    def act(self,state,training=True):
        eps=self.epsilon() if training else 0.0
        if training and random.random()<eps: return random.randrange(6),eps
        with torch.no_grad():
            x=torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(1).item()),eps
    def learn(self):
        if len(self.buffer)<max(self.cfg.warmup,self.cfg.batch_size): return None
        b=self.buffer.sample(self.cfg.batch_size)
        s=torch.tensor(np.array([x.state for x in b]),dtype=torch.float32,device=self.device)
        a=torch.tensor([x.action for x in b],dtype=torch.long,device=self.device).unsqueeze(1)
        r=torch.tensor([x.reward for x in b],dtype=torch.float32,device=self.device)
        ns=torch.tensor(np.array([x.next_state for x in b]),dtype=torch.float32,device=self.device)
        d=torch.tensor([x.done for x in b],dtype=torch.float32,device=self.device)
        q=self.online(s).gather(1,a).squeeze(1)
        with torch.no_grad():
            na=self.online(ns).argmax(1,keepdim=True)
            nq=self.target(ns).gather(1,na).squeeze(1)
            y=r+self.cfg.gamma*nq*(1-d)
        loss=nn.functional.smooth_l1_loss(q,y)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),self.cfg.grad_clip); self.opt.step()
        self.steps+=1
        if self.steps%self.cfg.target_update==0: self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

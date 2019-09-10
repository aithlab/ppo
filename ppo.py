# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:09:53 2019

@author: Taehwan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

from torch.distributions import Normal

class PPO(nn.Module):
  def __init__(self, state_dim, eps=0.2, gamma=0.98):
    super(PPO, self).__init__()
    self.eps = eps
    self.gamma = gamma
    
    self.fc1 = nn.Linear(state_dim, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 1)
    self.fc4 = nn.Linear(64,1)
    
    self.fc5 = nn.Linear(state_dim, 64)
    self.fc6 = nn.Linear(64, 64)
    self.fc7 = nn.Linear(64,1)
    
    self.buffer = {'s':[], 'a':[], 'next_s':[], 'r':[], 'v':[]}
    
  def forward(self, x):
    actor = torch.tanh(self.fc1(x))
    actor = torch.tanh(self.fc2(actor))
    mean = self.mean(actor)
    logvar = self.logvar(actor)
    self.pi = Normal(mean, (logvar/2).exp())

#    action = pi_dist.rsample() # reparameterized sample(mean + esp*std)
    
    self.v = nn.Sequential(self.fc5, 
                           nn.Tanh(),
                           self.fc6,
                           nn.Tanh(),
                           self.fc7)
    
    return self.pi, self.v
    
  def loss(self):
    prob_ratio = self.pi.log_prob(self.buffer['a'])#TODO:
    
    v_target = self.buffer['r'] + self.gamma*self.v(self.buffer['next_state'])
    delta = v_target - self.v(self.buffer['state'])
    
    advantage_fcn = advantage_function #TODO:
    
    first_tem = prob_ratio*advantage_fcn # loss_cpi(conservative policy iteration)
    second_term = self.clip_by_value(prob_ratio)*advantage_fcn
    loss_clip = torch.min(first_term, second_term)
    
    
    loss_vf = (self.buffer['v'] - self.buffer['r'])**2 # squared error loss: (v(s_t) - v_target)**2
    
    loss = loss_clip - loss_vf
  
  def buffer_update(self, s, a, next_s, r, v):
    self.buffer['s'].append(torch.from_numpy(s).float())
    self.buffer['a'].append(torch.from_numpy(a).float())
    self.buffer['next_s'].append(torch.from_numpy(next_s).float())
    self.buffer['r'].append(torch.from_numpy(np.array([r])).float())
    self.buffer['v'].append(v)
    
  def buffer_reset(self):
    self.buffer = {'s':[], 'a':[], 'next_s':[], 'r':[], 'v':[]}
    
  def advantage_function(self, x):
    pass
  
  def clip_by_value(self, x):
    return x.clamp(1-self.eps, 1+self.eps) # clamp(min, max)
  
env = gym.make('Pendulum-v0')

state_dim = env.observation_space.shape[0]
model = PPO(state_dim)

N_epoch = 20
Horizon = 128
for epoch in range(N_epoch):
  state = env.reset()
  for _ in range(Horizon):
    env.render()
    action_dist, v = model(torch.tensor(state, dtype=torch.float32))
    action = action_dist.rsample().detach().numpy()
    
    next_state, reward, done, info = env.step(action)
    model.buffer_update(state, action, next_state, reward, v)
    state = next_state
    if done:
      break
    
  model.buffer_reset()

env.close()

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:25:32 2019

@author: Taehwan
"""

import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
from buffer import Buffer
from ppo import PPO
from utils import NormalizedReward
SEED = 1234

N_epoch = 500
T = 200 #Peudulum environment has 200 max episode steps
gamma = 0.99
lambda_ = 0.95

env = NormalizedReward(gym.make('Pendulum-v0'))
torch.manual_seed(SEED)
env.seed(SEED)
np.random.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
ppo = PPO(state_dim, action_dim)

buffer = Buffer(T)
params = [p for p in ppo.model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0003)

epoch_r = []
for epoch in range(N_epoch):
  state, T_r, done, t = env.reset(), 0, False, 1
  state = Variable(torch.tensor(state, dtype=torch.float32))
  while not done: # Collect T timesteps of data
    env.render()
    
    action_dist, v_s_t = ppo(state)
    action = action_dist.rsample() #reparameterized sample
    entropy = action_dist.entropy()
    log_prob = action_dist.log_prob(action)
    
    next_state, reward, done, info = env.step(action.detach().numpy()) #numpy, tensor, bool, dict
    next_state = Variable(torch.tensor(next_state, dtype=torch.float32))
    T_r += reward
    t += 1
    if t > T:
      done = True
    reward = torch.as_tensor([reward])
    buffer.update(state, action, next_state, reward, log_prob, entropy)
    
    state = next_state
    
  epoch_r.append(T_r)
  ppo.update(buffer, optimizer)
  print('[%3d] Reward: %.3f'%(epoch, T_r))

plt.plot(epoch_r)
env.close()


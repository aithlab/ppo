# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:25:32 2019

@author: Taehwan
"""

import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from gym import RewardWrapper
from torch.autograd import Variable
from ppo import PPO
SEED = 1234

class NormalizeReward(RewardWrapper):
  def reward(self, reward):
    # th=[-3.14, 3.14], thdot = [-8,8], u = [-2,2]
    th = 3.14
    thdot = 8
    u = 2
    max_val = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
    
    return reward / max_val
  
  def angle_normalize(self, x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

env = NormalizeReward(gym.make('Pendulum-v0'))
torch.manual_seed(SEED)
env.seed(SEED)
np.random.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
model = PPO(state_dim, action_dim)

N_epoch = 500
T = 4000 #2048
gamma = 0.99
lambda_ = 0.95

epoch_r = []
for epoch in range(N_epoch):
  state, T_r = env.reset(), 0
  for _ in range(T): # Collect T timesteps of data
    env.render()
    action_dist, v_s_t = model(Variable(torch.tensor(state, dtype=torch.float32)))
    action = action_dist.rsample() #reparameterized sample
    entropy = action_dist.entropy()
    log_prob = action_dist.log_prob(action)
    
    next_state, reward, done, info = env.step(action.detach().numpy()) #numpy, tensor, bool, dict
    model.buffer_update(state, action, next_state, reward, log_prob, entropy)
    
    state = next_state
    T_r += reward
    
    if done:
      break
    
  epoch_r.append(T_r)
  model.update()
  print('[%3d] Reward: %.3f'%(epoch, T_r))

plt.plot(epoch_r)
env.close()


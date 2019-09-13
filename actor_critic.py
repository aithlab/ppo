#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:15:03 2019

@author: taehwan
"""

import torch
import torch.nn as nn

from torch.distributions import Normal

class ActorCritic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(ActorCritic, self).__init__()
    
    self.affine1 = nn.Linear(state_dim, 64)
    self.affine2 = nn.Linear(64, 64)
    self.affine3 = nn.Linear(64, action_dim)
    self.affine4 = nn.Linear(64, action_dim)
    
    self.affine5 = nn.Linear(state_dim, 64)
    self.affine6 = nn.Linear(64, 64)
    self.affine7 = nn.Linear(64,1)
    
  def forward(self, x):
    actor = torch.tanh(self.affine1(x))
    actor = torch.tanh(self.affine2(actor))
    mean = torch.tanh(self.affine3(actor))
    logvar = torch.tanh(self.affine4(actor))
    self.pi = Normal(mean, (logvar*0.5).exp())
    
    critic = torch.tanh(self.affine5(x))
    critic = torch.tanh(self.affine6(critic))
    self.v = self.affine7(critic)
    
    return self.pi, self.v


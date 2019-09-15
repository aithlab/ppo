#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:25:52 2019

@author: taehwan
"""
import torch

class Buffer():
  def __init__(self, K):
    self.K = K    
    self.reset()
    
  def update(self, s, a, next_s, r, log_prob, entropy):
    for key, value in zip(self.data.keys(), (s,a,next_s,r,log_prob,entropy)):
      self.data[key].append(value)
    self.current_length += 1
    
    if self.current_length > self.K:
      for key in self.data.keys():
        self.data[key].pop(0)
  
  def get_data(self, batch_size):
    n_data = len(self.data['s'])
    idxs = torch.randperm(n_data)
    
    self.buffer2torch(self.data)
     
    for start_pt in range(0,n_data,batch_size):
      state = self.data['s'][idxs][start_pt:start_pt+batch_size]
      action = self.data['a'][idxs][start_pt:start_pt+batch_size]
      next_s = self.data['next_s'][idxs][start_pt:start_pt+batch_size]
      reward = self.data['r'][idxs][start_pt:start_pt+batch_size]
      log_prob = self.data['log_prob'][idxs][start_pt:start_pt+batch_size]
      entropy = self.data['entropy'][idxs][start_pt:start_pt+batch_size]
      advantage = self.data['advantage'][idxs][start_pt:start_pt+batch_size]
      yield state, action, next_s, reward, log_prob, entropy, advantage
    
  def buffer2torch(self, buffer):
    for key in buffer.keys():
      if type(buffer[key]) is not torch.Tensor:
        buffer[key] = torch.stack(buffer[key], dim=0)
  
  def reset(self,):
    self.data = {'s':[], 'a':[], 'next_s':[], 'r': [], 
                 'log_prob':[], 'entropy':[]}
    self.current_length = 0
    
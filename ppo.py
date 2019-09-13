# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:09:53 2019

@author: Taehwan
"""

import torch
import torch.nn as nn

from actor_critic import ActorCritic

class PPO(nn.Module):
  def __init__(self, state_dim, action_dim, eps=0.2, gamma=0.99, lambda_=0.95, 
               K_epoch=80, batch_size=64):
    super(PPO, self).__init__()
    self.eps = eps
    self.gamma = gamma
    self.lambda_ = lambda_
    self.K_epoch = K_epoch
    self.batch_size = batch_size
 
    self.model = ActorCritic(state_dim, action_dim)
    self.model_old = ActorCritic(state_dim, action_dim)
    for param in self.model_old.parameters():
      param.requires_grad = False
    self.copy_weights()
    
    self.buffer_reset()
    params = [p for p in self.model.parameters() if p.requires_grad]
    self.optimizer = torch.optim.Adam(params, lr=0.0003)

    
  def forward(self, x):
    self.pi, self.v = self.model_old(x)
    
    return self.pi, self.v
    
  def copy_weights(self):
    self.model_old.load_state_dict(self.model.state_dict())
    
  def update(self):
    self.model.train()
    self.model_old.eval()
    self.advantage_fcn()

    batch_loss, batch_clip_loss, batch_vf_loss = [],[],[]
    for epoch in range(self.K_epoch):
      for state, action, next_s, reward, log_prob_old, entropy, advantage in self.from_buffer(self.batch_size):  
        pi, v = self.model(state)
        log_prob_pi = pi.log_prob(action)
        
        prob_ratio = torch.exp(log_prob_pi - log_prob_old)
            
        first_term = prob_ratio*advantage 
        second_term = self.clip_by_value(prob_ratio)*advantage
        loss_clip = (torch.min(first_term, second_term)).mean()
        
        _, v_next = self.model_old(next_s)
        v_target = reward + self.gamma*v_next
        loss_vf = ((v - v_target)**2).mean() # squared error loss: (v(s_t) - v_target)**2
        
        loss = -(loss_clip - loss_vf)
#        loss = -(loss_clip - 0.5*loss_vf + 0.01*entropy.mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        batch_loss.append(loss.detach().numpy())
        batch_clip_loss.append(loss_clip.detach().numpy())
        batch_vf_loss.append(loss_vf.detach().numpy())

    self.copy_weights()
    self.buffer_reset()
  
  def from_buffer(self, batch_size):
    n_data = len(self.buffer['s'])
    idxs = torch.randperm(n_data)
    
    for start_pt in range(0,n_data,batch_size):
      state = self.buffer['s'][idxs][start_pt:start_pt+batch_size]
      action = self.buffer['a'][idxs][start_pt:start_pt+batch_size]
      next_s = self.buffer['next_s'][idxs][start_pt:start_pt+batch_size]
      reward = self.buffer['r'][idxs][start_pt:start_pt+batch_size]
      log_prob = self.buffer['log_prob'][idxs][start_pt:start_pt+batch_size]
      entropy = self.buffer['entropy'][idxs][start_pt:start_pt+batch_size]
      advantage = self.buffer['advantage'][idxs][start_pt:start_pt+batch_size]
      yield state, action.unsqueeze(1), next_s, reward, log_prob.unsqueeze(1), entropy.unsqueeze(1), advantage
      
  def advantage_fcn(self, normalize=True):
    _, v_st1 = self.model(torch.as_tensor(self.buffer['next_s'], dtype=torch.float))
    _, v_s = self.model(torch.as_tensor(self.buffer['s'], dtype=torch.float))
    deltas = torch.as_tensor(self.buffer['r']) + self.gamma*v_st1 - v_s
    
    advantage,temp = [],0
    idxs = torch.tensor(range(len(deltas)-1,-1,-1)) #reverse
    reverse_deltas = deltas.index_select(0,idxs)
    for delta_t in reverse_deltas:
      temp = delta_t + self.lambda_*self.gamma*temp
      advantage.append(temp)
    
    advantage = torch.as_tensor(advantage[::-1]) #re-reverse
    if normalize:
      advantage = (advantage - advantage.mean()) / advantage.std()
    
    self.buffer['advantage'] = advantage.unsqueeze(1)
    self.buffer2torch()
  
  def buffer2torch(self):
    for key in self.buffer.keys():
      self.buffer[key] = torch.as_tensor(self.buffer[key], dtype=torch.float)
  
  def buffer_update(self, s, a, next_s, r, log_prob, entropy):
    self.buffer['s'].append(s) #array
    self.buffer['a'].append(a) #array
    self.buffer['next_s'].append(next_s) #array
    self.buffer['r'].append([r]) #tensor
    self.buffer['log_prob'].append(log_prob) #tensor
    self.buffer['entropy'].append(entropy)
    
  def buffer_reset(self):
    self.buffer = {'s':[], 'a':[], 'next_s':[], 'r':[], 'log_prob':[], 
                   'entropy':[], 'advantage':[]}
  
  def clip_by_value(self, x):
    return x.clamp(1-self.eps, 1+self.eps) # clamp(min, max)

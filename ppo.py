# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:09:53 2019

@author: Taehwan
"""

import torch
import torch.nn as nn

from actor_critic import ActorCritic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
  def forward(self, x):
    self.pi, self.v = self.model_old(x)
    
    return self.pi, self.v
    
  def copy_weights(self):
    self.model_old.load_state_dict(self.model.state_dict())
    
  def update(self, buffer, optimizer):
    self.model.train()
    self.model_old.eval()
    self.advantage_fcn(buffer.data)

    batch_loss, batch_clip_loss, batch_vf_loss = [],[],[]
    for epoch in range(self.K_epoch):
      for state, action, next_s, reward, log_prob_old, entropy, advantage in buffer.get_data(self.batch_size):  
        pi, v = self.model(state)
        log_prob_pi = pi.log_prob(action)
        
        prob_ratio = torch.exp(log_prob_pi - log_prob_old)
            
        first_term = prob_ratio*advantage 
        second_term = self.clip_by_value(prob_ratio)*advantage
        loss_clip = (torch.min(first_term, second_term)).mean()
        
        _, v_next = self.model_old(next_s)
        v_target = reward + self.gamma*v_next
        loss_vf = ((v - v_target)**2).mean() # squared error loss: (v(s_t) - v_target)**2
        
        loss = -(loss_clip - loss_vf) #-(loss_clip - 0.5*loss_vf + 0.01*entropy.mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.detach().numpy())
        batch_clip_loss.append(loss_clip.detach().numpy())
        batch_vf_loss.append(loss_vf.detach().numpy())

    self.copy_weights()
    buffer.reset()
      
  def advantage_fcn(self, buffer, normalize=True):
    _, v_st1 = self.model(torch.stack(buffer['next_s']))
    _, v_s = self.model(torch.stack(buffer['s']))
    deltas = torch.stack(buffer['r']) + self.gamma*v_st1 - v_s
    
    advantage,temp = [],0
    idxs = torch.tensor(range(len(deltas)-1,-1,-1)) #reverse
    reverse_deltas = deltas.index_select(0,idxs)
    for delta_t in reverse_deltas:
      temp = delta_t + self.lambda_*self.gamma*temp
      advantage.append(temp)
    
    advantage = torch.as_tensor(advantage[::-1]) #re-reverse
    if normalize:
      advantage = (advantage - advantage.mean()) / advantage.std()
    
    buffer['advantage'] = advantage.unsqueeze(1)
  
  def clip_by_value(self, x):
    return x.clamp(1-self.eps, 1+self.eps) # clamp(min, max)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 01:08:09 2019

@author: taehwan
"""
import numpy as np

from gym import RewardWrapper

class NormalizedReward(RewardWrapper):
  def reward(self, reward):
    # th=[-3.14, 3.14], thdot = [-8,8], u = [-2,2]
    th,thdot,u = 3.14,8,2
    max_val = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
    
    return reward / (max_val+1e-4)
  
  def angle_normalize(self, x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
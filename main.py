# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:25:32 2019

@author: Taehwan
"""

import gym

env = gym.make('Pendulum-v0')
env.seed(1234)

state = env.reset() #[con(theta), sin(theta), theta_dot]
state, reward, done, _ = env.step([.7])
env.render()
env.close()

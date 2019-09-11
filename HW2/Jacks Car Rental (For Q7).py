#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import random


# In[ ]:


class DP:
    def __init__(self, size, actions, policy, discount):
        self.size = size
        self.actions = actions
        self.discount = discount
        self.policy = policy
    
    def printPolicy(self):
        for i in self.policy:
            print(self.policy[i])
        print()
        
    def getNextState(self, state, action):
        if(isterminal(state, self.size)):
            return state, 0
        else:
            next_state = tuple(np.array(state) + np.array(action))
            if(next_state[0] < self.size and next_state[1] < self.size and next_state[0] >= 0 and next_state[1] >= 0):
                return next_state, -1
            else:
                return state, -1
        
    def policy_iteration(self):
        new_v =  np.zeros((self.size, self.size))
        while(True):
            #policy evaluation
            values = np.zeros((self.size, self.size))
            iteration = 0
            while(True):
                new_v = np.copy(values)
                delta = 0
                for i in range(self.size):
                    for j in range(self.size):
                        state = (i,j)
                        v = 0
                        for a in range(len(self.actions)):
                            new_state, reward = self.getNextState(state, self.actions[a])
                            v += self.policy[state][a]*(reward + self.discount*values[new_state[0], new_state[1]])
                        delta = max(delta, np.abs(v - new_v[i,j]))
                        new_v[i,j] = v
                values = new_v
                
                if(delta < 1e-5):
                    break
            
            #policy improvement
            policy_stable = True
            for i in range(self.size):
                for j in range(self.size):
                    action_values = np.zeros(len(self.actions))
                    state = (i,j)
                    for a in range(len(self.actions)):
                        new_state , reward = self.getNextState(state, self.actions[a])
                        action_values[a] += (reward + self.discount*new_v[new_state[0], new_state[1]])
                    old_action = np.argmax(self.policy[state])
                    new_action = np.argmax(action_values)
    
                    if(old_action != new_action):
                        policy_stable = False
                    self.policy[state] = np.eye(4)[new_action]

            if(policy_stable):
                print()
                return self.policy, new_v
                


# In[ ]:





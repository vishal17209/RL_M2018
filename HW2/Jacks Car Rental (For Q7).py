#!/usr/bin/env python
# coding: utf-8

# In[91]:

import numpy as np
import matplotlib.pyplot as plt
import random
from math import factorial


# In[92]:


#utitlity functions
def isterminal(state, size):
    if(state == (0,0) or state == (size-1,size-1)):
        return True
    return False

def poisson(n,lam):
    prob = ((lam**n)*np.exp(-lam))/factorial(n)
    return prob


# In[1]:


class DP:
    def __init__(self, size, actions, policy, discount):
        self.size = size
        self.actions = actions
        self.discount = discount
        self.policy = policy
        self.lambda_rental1 = 3
        self.lambda_rental2 = 4
        self.lambda_return1 = 3
        self.lambda_return2 = 2
    
    def printPolicy(self):
        for i in self.policy:
            print(self.policy[i])
        print()
        
    def getReturn(self, state, values, action):
        returns = -2*np.abs(action)
        
        carsAt1 = min(state[0] + action, self.size-1) 
        carsAt2 = min(state[1] - action, self.size-1)
        
        for rental1 in range(11):
            for rental2 in range(11):
                prob = poisson(rental1, self.lambda_rental1)*poisson(rental2, self.lambda_rental2)
                
                available1 = min(rental1, carsAt1)
                available2 = min(rental2, carsAt2)
                
                rewards = (available1  + available2) * 10
                
                for return1 in range(11):
                    for return2 in range(11):
                        prob *= poisson(return1, self.lambda_return1)*poisson(return2, self.lambda_return2)
                        
                        carAt1EOD = min(max(carsAt1 - rental1 + return1, 0), self.size-1)
                        carAt2EOD = min(max(carsAt2 - rental2 + return2, 0), self.size-1)

                        returns += prob*(rewards + self.discount*values[carAt1EOD, carAt2EOD])
        return returns  
    
    def getAllReturns(self, state, values):
        action_values = np.zeros(len(self.actions))
        for a in range(len(self.actions)):
            action_values[a] += self.getReturn(state, values, self.actions[a])
        return action_values
        
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
                            v+= self.getReturn(state, values, self.actions[a])
                        delta = max(delta, np.abs(v - new_v[i,j]))
                        new_v[i,j] = v
                values = new_v
                print(delta)
                if(delta < 1e-2):
                    break

            #policy improvement
            policy_stable = True
            for i in range(self.size):
                for j in range(self.size):
                    action_values = np.zeros(len(self.actions))
                    state = (i,j)
                    action_values = self.getAllReturns(state, values)
                    old_action = np.argmax(self.policy[state])
                    new_action = np.argmax(action_values)
    
                    if(old_action != new_action):
                        policy_stable = False
                    self.policy[state] = np.eye(4)[new_action]

            if(policy_stable):
                return self.policy, new_v
                


# In[94]:


max_cars = 20
max_car_moves = 5

fake_policy = {}

#taking positive actions as addition to location 1 and negative actions as addition to location 2
actions = []
for i in range(-5,6,1):
    actions.append(i)

for i in range(max_cars+1):
    for j in range(max_cars+1):
        fake_policy[(i,j)] = np.ones(len(actions))/len(actions)

dp = DP(max_cars+1, actions, fake_policy, 0.9)

policy, v = dp.policy_iteration()

for i in policy:
    print(policy)
print()

print(np.round(v))
print()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[77]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import random


# # Fig 3.2
# 
# Figure 3.2 is generated using Bellmans equations. Firstly using all the avaialble states we model a system of equations.
# The matrix A represents the coefficients of Vpi(s') and b represents the constant that comes when we rearrange Bellmans equations
# 
# # Fig 3.5
# 
# Figure 3.5 is generated using Non linear equations and these equations are estimated using iterations to get the final value matrix.

# In[78]:


'''
Class which is used as a template to implement MDPs
'''
class MDP:
    def __init__(self, size, policy, discount, actions):
        self.size = size
        self.policy = policy
        self.discount = discount
        self.actions = actions
        
    '''
    Function takes current state and action as input and gives
    the reward and next state as output according to gridworld
    '''
    def getNextState(self, state, action):
        if(np.array_equal(state, np.array([0,1]))):
            return np.array([4,1]), 10
        elif(np.array_equal(state, np.array([0,3]))):
            return np.array([2,3]), 5
        new_state = state + action
        new_i, new_j = new_state
        if(new_i < 0 or new_i > self.size-1 or new_j < 0 or new_j > self.size-1):
            new_state = state
            return new_state, -1
        return new_state, 0
              
    '''
    Function to generate Fig3.2
    '''
    def generateFig32(self):
        A = np.zeros((self.size**2, self.size**2))
        b = np.zeros(self.size**2)
            
        #matrix to index A and b
        m = [[0 for i in range(self.size)] for j in range(self.size)]
        c = 0
        for i in range(self.size):
            for j in range(self.size):
                m[i][j] = c
                c+=1
        
        for i in range(self.size):
            for j in range(self.size):
                for a in range(len(self.actions)):
                    '''
                    Getting the next state from the current state for all actions and then constructing the
                    matrix A which is the coefficient of Vpi(s') 
                    And matrix b which is the constant that is policy*step_reward
                    '''
                    [new_i, new_j] , rewards = self.getNextState(np.array([i,j]), self.actions[a])
                    A[m[i][j], m[new_i][new_j]] += self.policy*self.discount
                    b[m[i][j]] += self.policy*rewards
                A[m[i][j],m[i][j]] = A[m[i][j],m[i][j]]-1
        
        #solving the system of equations using linalg
        X = np.linalg.solve(A,b)
        return np.round(np.reshape(X*(-1), (5,5)),1)
    
    '''
    Function to generate Fig3.5
    '''
    def generateFig35(self):
        value_func = np.zeros((self.size, self.size))
        updated_values = np.zeros_like(value_func)
        error = 1000
        
        '''
        Iterating over all states and actions to get the value function until the function has converged
        assuming the equation is estimated properly when the error is <1e-4
        '''
        while(error > 1e-4):
            updated_values = np.zeros_like(value_func)
            #Iterating over all states
            for i in range(self.size):
                for j in range(self.size):
                    value_log = []
                    for a in self.actions:
                        #getting new states using current state and action
                        [new_i, new_j], reward = self.getNextState(np.array([i,j]), a)
                        value_log.append(reward + self.discount*value_func[new_i, new_j])
                    
                    #updating the value function with the greedy choice
                    updated_values[i,j] = np.max(value_log)
            
            #calculating error and udpating the main function
            error = np.sum(np.abs(value_func - updated_values))
            value_func = updated_values
            # print(error)
        return np.round(updated_values, 1)
            


# In[79]:


mdp = MDP(5, 0.25, 0.9, np.array([[0,-1],[0,1],[1,0],[-1,0]]))

valueFig32 = mdp.generateFig32()

print("Fig 3.2: ")
for i in range(5):
    for j in range(5):
        print(valueFig32[i,j], end=" ")
    print()


# In[80]:


valueFig35 = mdp.generateFig35()

print("Fig 3.5: ")
for i in range(5):
    for j in range(5):
        print(valueFig35[i,j], end=" ")
    print()


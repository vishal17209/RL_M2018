#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


#utitlity functions
def isterminal(state, size):
    if(state == (0,0) or state == (size-1,size-1)):
        return True
    return False


# # Policy Iteration and Bug Fix
# 
# Policy Iteration is method to estimate optimal policy for a given MDP. 
# steps in the algorithm are:
# 
# while(policy not stable){  
#      Policy_Evaluation();  
#      Policy_Improvement();  
# }  
# 
# Since we are guranteed that every time we improve our policy our value function also improves we should get the optimal policy when the new policy is same as the old policy we can stop iterations.
# 
# As given in the exercise 4.4 of S+B the pseudocode should enter an infinite loop if the policy continuously switches between two or more policies.
# Which implies that at every step the code chosing two equally optimal actions and thus a fix should be compare the maximum of optimal value function as all optimal policies will lead to the same optimal value function.
# This bug fix is marked by comments in the code in the Policy Improvement section of PI.

# In[3]:


'''
Class which is used as a template to implement DP
'''
class DP:
    #Constructor
    def __init__(self, size, actions, policy, discount):
        self.size = size
        self.actions = actions
        self.discount = discount
        self.policy = policy
    
    #function to print the policy variable of an object
    def printPolicy(self):
        for i in self.policy:
            print(self.policy[i])
        print()
        
    '''
    This function takes current state - state and the action to be performed - action as inputs 
    and gives the next state and reward according to gridworld problem
    '''
    def getNextState(self, state, action):
        if(isterminal(state, self.size)):
            return state, 0
        else:
            next_state = tuple(np.array(state) + np.array(action))
            if(next_state[0] < self.size and next_state[1] < self.size and next_state[0] >= 0 and next_state[1] >= 0):
                return next_state, -1
            else:
                return state, -1
        
    '''
    Function perform Policy iteration
    '''
    def policy_iteration(self):
        
        #lists to log the value functions and the policy
        value_log = []
        policy_log = []
        
        while(True):
            #policy evaluation
            values = np.zeros((self.size, self.size))
            
            #for logging data
            iteration = 0
            value_log.append([])
            
            while(True):
                new_v = np.copy(values)
                delta = 0
                
                '''
                Iterating over all the states and then over all actions to get the matrix of value functions
                and then repeating the process to evaluate the optimal value function
                '''
                for i in range(self.size):
                    for j in range(self.size):
                        state = (i,j)
                        v = 0
                        for a in range(len(self.actions)):
                            #getting new states using current state and action
                            new_state, reward = self.getNextState(state, self.actions[a])
                            v += self.policy[state][a]*(reward + self.discount*values[new_state[0], new_state[1]])
                            
                        #Calcuating the difference between new and old values
                        delta = max(delta, np.abs(v - new_v[i,j]))
                        new_v[i,j] = v
                        
                #updating value function
                values = new_v
                
                #logging data
                value_log[iteration].append(values)
                
                #Break when converged
                if(delta < 1e-5):
                    break
            
            #policy improvement
            policy_stable = True
            
            '''
            Iterating over all the states and then calculating action values using the old policy and the new policy
            obtained by Policy evaluation and then stopping if both the policy and the value function remains the same(fix for bug in ex4.4)
            '''
            for i in range(self.size):
                for j in range(self.size):
                    action_values = np.zeros(len(self.actions))
                    state = (i,j)
                    for a in range(len(self.actions)):
                        #getting new states using current state and action
                        new_state , reward = self.getNextState(state, self.actions[a])
                        action_values[a] += (reward + self.discount*values[new_state[0], new_state[1]])
                        
                    #Getting the old and new action using argmax
                    old_action = np.argmax(self.policy[state])
                    new_action = np.argmax(action_values)
                        
                    #Bug Fix
                    #Getting the old and new optimal values
                    old_value = np.max(self.policy[state])
                    new_value = np.max(action_values)
            
                    # (Bug Fix)Stopping if the policy and value function didnt change as they will always become better otherwise
                    if(old_action != new_action and old_value != new_value):
                        policy_stable = False
                    
                    #Updating policy
                    temp = np.zeros(len(self.actions))
                    temp[new_action] = 1
                    self.policy[state] = temp 
                    
                    policy_log.append(self.policy)
            
            #returning policy when its stable
            if(policy_stable):
                return self.policy, policy_log, value_log
                        
    '''
    Function to perform Value Iteration
    '''
    def value_iteration(self):
        
        #for logging data
        value_log = []
        values = np.zeros((self.size,self.size))
        
        while(True):
            delta = 0
            new_v = np.copy(values)
            
            '''
            Iterating over all states and actions and calculating value function and then
            updating the old value function with the optimal value function
            '''
            for i in range(self.size):
                for j in range(self.size):
                    state = (i,j)
                    v = new_v[i,j]
                    actions_values = np.zeros(len(self.actions))
                    for a in range(len(self.actions)):
                        #getting new states using current state and action
                        new_state, reward = self.getNextState(state, self.actions[a])
                        actions_values[a] += (reward + self.discount*values[new_state[0], new_state[1]])
                    
                    #Updating with the new optimal value function
                    new_v[i,j] = np.max(actions_values)
                    delta = max(delta, np.abs(new_v[i,j] - v))
                    
            #updating and logging values
            values = new_v
            value_log.append(values)
            if(delta < 1e-4):
                break
        
        #Calculating the optimal policy using the optimal value function obtained above
        policy = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                state = (i,j)
                action_values = np.zeros(len(self.actions))
                for a in range(len(self.actions)):
                    new_state, reward = self.getNextState(state, self.actions[a])
                    action_values[a] += (reward + self.discount*values[new_state[0], new_state[1]])
                policy[i,j] = np.argmax(action_values)
        return policy, value_log
            


# In[4]:


fake_policy = {}

for i in range(4):
    for j in range(4):
        fake_policy[(i,j)] = np.ones(4)/4
        
dp = DP(4, [(-1,0),(0,1),(1,0),(0,-1)], fake_policy, 1)

policy, pl, vl = dp.policy_iteration()

print("Optimal Policy (0 = up, 1 = right, 2 = left, 3 = left)")
temp = []
for i in range(4):
    for j in range(4):
        print(np.argmax(policy[(i,j)]), end=" ")
    print()
print()

print("Few steps of Policy Evaluation for Iteration 1: ")
for i in range(3):
    print("Evaluation iteration number: ", i)
    print(vl[0][i])
    print()
    
print("Few Steps of Policy Improvement: ")
for k in range(3):
    print("For iteration ", k)
    temp = []
    for i in range(4):
        for j in range(4):
            print(np.argmax(pl[k][i,j]), end=" ")
        print()
    print()


# In[5]:


dp_v = DP(4, [(-1,0),(0,1),(1,0),(0,-1)], fake_policy, 1)
policy_v , vvl = dp_v.value_iteration()

print("Optimal Policy (0 = up, 1 = right, 2 = left, 3 = left)")

for i in range(4):
    for j in range(4):
        print(int(policy_v[i,j]), end=" ")
    print()
print()

print("Few Steps of Values Iteration")
for i in range(3):
    print("For iteration ", i)
    print(vvl[i])
    print()


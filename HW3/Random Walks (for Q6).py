#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn
from tqdm import tqdm                  #this is used to detect infinite loops for debugging


# In[16]:


'''
Class that is used to generate the environment for Random Walks problem from the book
'''
class Random_Walks:
    '''
    Constructor sets the parameters required to generate given environment
    '''
    def __init__(self):
        self.actions = [0,1] #0->left, 1->right
        self.true_values = np.zeros(7)
        self.true_values[1:6] = np.arange(1,6)/6
        self.true_values[6] = 1
        
    '''
    Function for Temporal Difference. It takes values that is the value function as input and alpha and performs
    TD learning on the given environment with gamma = 1 values is updated as numpy arrays mutable
    '''
    def TD(self, values, alpha = 0.1):
        state = 3
        series = [state]
        all_rewards = [0]
        
        while(True):
            prev_state = state
            if(random.random() < 0.5):
                state -= 1
            else:
                state += 1
            
            step_reward = 0
            series.append(state)
            
            values[prev_state] += alpha*(step_reward + values[state] - values[prev_state])
            if(state == 6 or state == 0):
                break
            all_rewards.append(step_reward)
        return series, all_rewards
    
    '''
    Function to perform Monte Carlo simulation on the given environment. It takes values that is the value function as input and alpha and performs
    MC on the given environment with gamma = 1 values is updated as numpy arrays mutable
    '''
    def MC(self, values, alpha=0.1):
        state = 3
        series = [state]
        
        while(True):
            if(random.random() < 0.5):
                state -= 1
            else:
                state += 1
            
            series.append(state)
            if(state == 6):
                returns = 1
                break
            elif(state == 0):
                returns = 0
                break
            
        for state in series[:-1]:
            values[state] += alpha*(returns - values[state])
                
        return series, [returns for i in range(len(series)-1)]
    
    '''
    Creates the left side plot in example 6.2
    '''
    def leftplot(self):
        #initialising state value function
        values = np.zeros(7)
        values[1:6] = 0.5
        values[6] = 1
        
        plt.figure(figsize=(20,10))
        
        #running episodes and sampling state values from the ones of interest
        for epi in range(101):
            if(epi in [0,1,10,100]):
                plt.plot([1,2,3,4,5], values[1:-1], label=str(epi) + ' episodes')
            series, rewards = self.TD(values)
        
        plt.plot([1,2,3,4,5], self.true_values[1:-1], label='true_values')
        plt.xlabel('state')
        plt.ylabel('estimated values')
        plt.xticks([1,2,3,4,5])
        plt.legend(prop={"size":20})
        
    '''
    Created the right plot in example 6.2
    '''
    def rightplot(self):
        #array of all alphas
        alpha = [0.15,0.1,0.05,0.01, 0.02, 0.03, 0.04]
        plt.figure(figsize=(20,10))
        for i in range(len(alpha)):
            total_error = np.zeros(100)
            '''
            first three alphas are for TD. Then we set the title and linestyle accordingly and perform TD
            and calculate mean square errors averaged over all the states
            '''
            if(i < 3):
                title = "TD"
                line = "solid"
                for j in range(100):
                    #initisalise variables
                    run_errors = []
                    values = np.zeros(7)
                    values[1:6] = 0.5
                    values[6] = 1
                    
                    for k in range(100):
                        run_errors.append(np.sqrt(np.sum((self.true_values - values)**2)/5))
                        series, rewards = self.TD(values, alpha[i])
                    total_error += np.array(run_errors)
            else:
                '''
                after first three alphas are for MC. Then we set the title and linestyle accordingly and perform TD
                and calculate mean square errors averaged over all the states
                '''
                title = "MC"
                line = "dashdot"
                for j in range(100):
                    #initisalise variables
                    run_errors = []
                    values = np.zeros(7)
                    values[1:6] = 0.5
                    values[6] = 1
                    
                    for k in range(100):
                        run_errors.append(np.sqrt(np.sum((self.true_values - values)**2)/5))
                        series, rewards = self.MC(values, alpha[i])
                    total_error += np.array(run_errors)
            total_error/=100
            plt.plot(total_error, linestyle=line, label=title + 'alpha = ' + str(alpha[i]))
        plt.xlabel("episodes")
        plt.ylabel("RMS error")
        plt.legend(prop={"size":20})


# In[17]:


randomwalks = Random_Walks()
randomwalks.leftplot()


# In[18]:


randomwalks.rightplot()


# In[ ]:





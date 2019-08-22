import numpy as np
import random

number_bandits = 2000
k = 10

q_true = np.random.normal(0,1,(number_bandits, k))
epsilons = [0, 0.1, 0.01]

for e in epsilons:
    q = np.zeros((number_bandits, k))
    n = np.zeros((number_bandits, k))
    qi = np.random.normal(q_true, 1)

    R_eps = []
    R_eps.append(0)
    R_eps.append(np.mean(qi))
    R_eps_opt = []

    for i in range(1,1001,1):
        R_temp = []
        for j in range(number_bandits):
            if(random.random() < e/10):
                p = np.random.randint(k)
            else:
                p = np.argmax(q[j]) 
            
            R = np.random.normal(q_true[i][j], 1)
            R_temp.append(R)
            n[i][j]+=1
            q[i][j] += (R-q[i][j])/n[i][j]

        R_avg = np.mean(R_temp)
        R_eps.append(R_avg)

print(qi)

# print(q_init)

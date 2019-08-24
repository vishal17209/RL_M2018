import numpy as np
import matplotlib.pyplot as plt
import random

number_bandits = 2000
k = 10

q_true = np.random.normal(0,1,(number_bandits, k))
epsilons = [0, 0.1, 0.01]
max_arms = np.argmax(q_true,1)

color = ['g','b','r']

fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

for e in epsilons:
    q = np.zeros((number_bandits, k))
    n = np.zeros((number_bandits, k))
    q_init = np.random.normal(q_true, 1)

    R_eps = []
    R_eps.append(0)
    R_eps.append(np.mean(q_init))
    R_eps_opt = []

    for i in range(2,1001,1):
        opt_arm_pull = 0
        R_temp = []
        for j in range(number_bandits):
            if(random.random() < e):
                p = np.random.randint(k)
            else:
                p = np.argmax(q[j]) 

            if(p == max_arms[j]):
                opt_arm_pull = opt_arm_pull + 1
            
            R = np.random.normal(q_true[j][p], 1)
            R_temp.append(R)
            n[j][p]+=1
            q[j][p] += (R-q[j][p])/n[j][p]

        R_avg = np.mean(R_temp)
        R_eps.append(R_avg)
        R_eps_opt.append(float(opt_arm_pull)/20)

    fig1.plot(range(0,1001,1), R_eps, color[epsilons.index(e)])
    fig2.plot(range(2,1001,1), R_eps_opt, color[epsilons.index(e)])

fig1.title.set_text('Average Reward')
fig2.title.set_text('Optimal Action')

fig2.set_ylim(0,100)

plt.show()

# print(q_init)
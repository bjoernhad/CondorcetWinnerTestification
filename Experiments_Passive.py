#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This file contains the experiments in the passive scenario CW testification 
scenario of [Haddenhorst2021].
In order to rerun these, simply execute this file.
The results are saved as a plot in "Change_RUCB_to_DTS_Hudry.png".

[Haddenhorst2021]: B. Haddenhorst, V. Bengs, J. Brandt and E.Hüllermeier, Testification of Condorcet Winners in Dueling Bandits, Proceedings of UAI, 2021
"""

import ReciprocalRelations as rr
import TestEnvironment as tenv
import TestingComponent as tc
import RankingAlgorithm as rankalg
import numpy as np
import matplotlib.pyplot as plt
import Regret as reg

def DTS(TE, h, gamma, n_arms, iterations, regret):
    """
This is an implementation of the Double Thompson Sampling algorithm from [Wu2016].

[Wu2016]: Huasen Wu and Xin Liu. Double Thompson sampling for dueling bandits. In Proceedings of Advances in Neural Information Processing Systems (NIPS), pages 649–657, 2016.
    """
    SymmTC = tc.Symmetric_TestingComponent(TE.N, TE.R, h=float(h), gamma=gamma)
    sampling_strategy = rankalg.PBMAB_DTS(n_arms)
    for t in range(iterations):
        if t % 500 == 0:
            print(t," iterations for DTS completed")
        [i,j] = sampling_strategy.getQuery()
        feedback = TE.pullArmPair(i,j)
        SymmTC.update(i,j,feedback)
        sampling_strategy.giveFeedback(feedback)
        regret.update_copeland_regret(i,j)
    return regret.copeland_regret

def RUCB(TE, h, gamma, n_arms,iterations, regret):
    """
This is an implementation of the RUCB algorithm from [Zoghi2014].

[Zoghi2014]: Masrour Zoghi, Shimon Whiteson, Remi Munos, and Maarten de Rijke. Relative upper confidence bound for the k-armed dueling bandit problem. In Proceedings of International Conference on Machine Learning (ICML), pages 10–18, 2014.
    """
    SymmTC = tc.Symmetric_TestingComponent(TE.N, TE.R, h=float(h), gamma=gamma)
    sampling_strategy = rankalg.PBMAB_RUCB(n_arms)
    for t in range(iterations):
        if t % 500 == 0:
            print(t," iterations for RUCB completed")
        [i,j] = sampling_strategy.getQuery()
        feedback = TE.pullArmPair(i,j)
        SymmTC.update(i,j,feedback)
        sampling_strategy.giveFeedback(feedback)
        regret.update_copeland_regret(i,j)
    return regret.copeland_regret

def change_from_RUCB_to_DTS(TE, h, gamma, n_arms, iterations, regret):
    """ 
    """
    SymmTC = tc.Symmetric_TestingComponent(TE.N, TE.R, h=float(h), gamma=gamma)
    sampling_strategy = rankalg.PBMAB_RUCB(n_arms)
    changed = False
    for t in range(iterations):
        if t % 500 == 0:
            print(t," iterations for RUCB->DTS completed")
        [i,j] = sampling_strategy.getQuery()
        feedback = TE.pullArmPair(i,j)
        SymmTC.update(i,j,feedback)
        sampling_strategy.giveFeedback(feedback)
        regret.update_copeland_regret(i,j)
        if SymmTC.TC() and not changed:
            if not SymmTC.DC():
                N=sampling_strategy.observedN
                R=sampling_strategy.observedR
                t= sampling_strategy.time
                sampling_strategy = rankalg.PBMAB_DTS(n_arms, time=t, N=N, R=R)
                changed = True
    return regret.copeland_regret

# Q_tennis
# more difficult matrix without CW from paper: Dueling BAndits beyond condorcet winners to general tounament solutions
# tennis tunier matrix (and slightly modified versions)
# Q = [[0.5,0.6,0.6,0.6,0.6,0.6,0.4,0.6],
#             [0.4,0.5,0.6,0.6,0.6,0.4,0.6,0.6],
#             [0.4,0.4,0.5,0.4,0.4,0.4,0.4,0.4],
#             [0.4,0.4,0.6,0.5,0.6,0.6,0.4,0.4],
#             [0.4,0.4,0.6,0.4,0.5,0.6,0.4,0.6],
#             [0.4,0.6,0.6,0.4,0.4,0.5,0.6,0.6],
#             [0.6,0.4,0.6,0.6,0.6,0.4,0.5,0.6],
#             [0.4,0.4,0.6,0.6,0.4,0.4,0.4,0.5]
#           ]
# Q = [[0.50, 0.40, 0.67, 0.60, 0.60, 0.83, 0.60, 0.73],
#             [0.60, 0.50, 0.60, 0.71, 0.67, 0.40, 0.40, 0.60],
#             [0.33, 0.40, 0.50, 0.37, 0.40, 0.38, 0.40, 0.20],
#             [0.40, 0.29, 0.63, 0.50, 0.71, 0.60, 0.17, 0.14],
#             [0.40, 0.33, 0.60, 0.29, 0.50, 0.75, 0.32, 0.60],
#             [0.17, 0.60, 0.62, 0.40, 0.25, 0.50, 0.29, 0.00],
#             [0.40, 0.60, 0.60, 0.83, 0.68, 0.71, 0.50, 0.60],
#             [0.27, 0.40, 0.80, 0.86, 0.40, 1.00, 0.40, 0.50]
#           ]
# Q= [[0.50, 0.47, 0.67, 0.53, 0.57, 0.83, 0.55, 0.73],
#             [0.53, 0.50, 0.57, 0.71, 0.67, 0.48, 0.43, 0.60],
#             [0.33, 0.43, 0.50, 0.37, 0.41, 0.38, 0.40, 0.20],
#             [0.47, 0.29, 0.63, 0.50, 0.71, 0.52, 0.17, 0.14],
#             [0.43, 0.33, 0.59, 0.29, 0.50, 0.75, 0.32, 0.58],
#             [0.17, 0.52, 0.62, 0.48, 0.25, 0.50, 0.29, 0.00],
#             [0.45, 0.57, 0.60, 0.83, 0.68, 0.71, 0.50, 0.52],
#             [0.27, 0.40, 0.80, 0.86, 0.42, 1.00, 0.48, 0.50]
#           ]
    
# The following is the Hudry Tournament, cf. also the following paper:
# [Ramamohan2016]: Siddartha Ramamohan, Arun Rajkumar, and Shivani Agarwal. Dueling bandits: Beyond Condorcet winners to general tournament solutions. In Proceedings of Advances in Neural Information Processing Systems (NIPS), pages 1253–1261, 2016.
Q = [[0.5, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
           [0.9, 0.5, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
           [0.9, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
           [0.9, 0.9, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
           [0.4, 0.9, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
           [0.4, 0.9, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
           [0.4, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
           [0.4, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9],
           [0.4, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9],
           [0.4, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9],
           [0.4, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9],
           [0.4, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9],
           [0.4, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
          ]

iterations = 6000 #before: 20000
h = 0.1
gamma = 0.1

n_arms = len(Q)
Q = np.array(Q)
Q = rr.ReciprocalRelation(Q)

r_rucb_tennis = reg.Regret(Q)
r_dts_tennis = reg.Regret(Q)
r_changing_tennis = reg.Regret(Q)

np.random.seed(1)
TE = tenv.TestEnvironment(Q)
r_changing_tennis = change_from_RUCB_to_DTS(TE, h, gamma, n_arms, iterations, r_changing_tennis)
TE = tenv.TestEnvironment(Q)
r_dts_tennis = DTS(TE, h, gamma, n_arms, iterations, r_dts_tennis)
TE = tenv.TestEnvironment(Q)
r_rucb_tennis = RUCB(TE, h, gamma, n_arms, iterations, r_rucb_tennis)

plt.plot(np.arange(iterations), r_changing_tennis, label="RUCB->DTS")
plt.plot(np.arange(iterations), r_rucb_tennis, label="RUCB")
plt.plot(np.arange(iterations), r_dts_tennis, label="DTS")
plt.xlabel("Iteration")
plt.ylabel("Copeland regret")
plt.legend(loc="lower right")
plt.savefig("Change_RUCB_to_DTS_Hudry.png",dpi=300)
plt.show()
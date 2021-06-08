#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains an implementation of the  SELECT algorithm from [Mohajer2017].

[Mohajer2017]: Soheil Mohajer, Changho Suh, and Adel Elmahdy. Active learning for top-k rank aggregation from noisy comparisons. In Proceedings of International Conference on Machine Learning (ICML), pages 2488–2497, 2017.

"""

import math
import numpy as np

def select(X, m, compare):
    # X: data to sort (in our case arms)
    # m: number of comparisons per pair... something like confidence but for all pairs
    itera = 0
    n = len(X)
    a = list(np.arange(n))
    if int(math.log(n, 2)) == math.log(n, 2):
        last_l = int(math.log(n, 2))
    else:
        last_l = int(math.log(n, 2)) + 1
    for l in range(last_l):
        if int(n / math.pow(2, l + 1)) == n / math.pow(2, l + 1):
            last_i = int(n / math.pow(2, l + 1))
        else:
            last_i = int(n / math.pow(2, l + 1)) + 1
        for i in range(last_i):
            T = 0
            if len(a) >= 2 * i + 2:
                if a[2 * i] == a[2 * i + 1]:
                    continue
            else:
                if a[2 * i] == a[2 * i - 1]:
                    continue
            for t in range(m):
                if len(a) > 2 * i + 1:
                    Y = compare(int(a[2 * i]), int(a[2 * i + 1]))
                else:
                    Y = compare(int(a[2 * i]), int(a[2 * i - 1]))
                itera += 1
                if Y:
                    T += 1
            if T >= m/2:
                a[i] = a[2*i]
            else:
                if len(a) > 2 * i + 1:
                    a[i] = a[2*i+1]
                else:
                    a[i] = a[2 * i - 1]
    return a[0], itera



###############################################################################
# The following lines are an example how SELECT may be executed.
# To execute them, add the two lines
#   import TestEnvironment as tenv
#   import ReciprocalRelations as rr
# to the beginning of this file.
###############################################################################
# np.random.seed(42)
# Q_CW = [[0.5, 0.6, 0.6, 0.4],
#         [0.4, 0.5, 0.6, 0.4],
#         [0.4, 0.4, 0.5, 0.4],
#         [0.6, 0.6, 0.6, 0.5]]
# Q_CW = np.array(Q_CW)
# M = Q_CW
# h = 0.01
# epsilon = 4
# m = math.floor((1+epsilon)*math.log(2)/2*math.log(math.log(len(M),2),2)/h)+1
# gamma = math.pow(math.log(len(M),2), (-1)*epsilon)
# #ms = [5, 10, 20, 50, 100, 200, 500, 1000, 5000]
# CWs = []
# #for m in ms:
# for i in range(10):
#     P = rr.ReciprocalRelation(Q=M, decimal_precision=decimal_precision)
#     TE = tenv.TestEnvironment(P)
#     CWs.append(select(np.arange(len(M)), m, TE.pullArmPair))
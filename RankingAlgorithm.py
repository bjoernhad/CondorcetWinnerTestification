#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains different ranking algorithms (=Sampling Strategies) for the 
Dueling Bandits Scenario.
""" 
 
import TestEnvironment as tenv
import ReciprocalRelations as rr
import numpy as np
import math
import random
from scipy.stats import beta

###############################################################################
# Class: PBMAB_algorithm 
###############################################################################
class PBMAB_algorithm(object):
    """
This is the superclass "PBMAB_algorithm" Do NOT create an object of this class 
itself, use one of the subclasses instead. To see these, run

>> print([cls.__name__ for cls in PBMAB_algorithm.__subclasses__()])

Each subclass has to contain the methods "getQuery" and "giveFeedback".
    """
    def __init__(self,m):
        assert False, "Do NOT create an object of the superclass `PBMAB_algorithm` itself!"
    
    def getQuery(self):
        """Every PBMAB_algorithm needs a method `getQuery`. Note: Do NOT create an object of the superclass `PBMAB_algorithm` itself!"""
        assert False, "Every PBMAB_algorithm needs a method `getQuery`. Note: Do NOT create an object of the superclass `PBMAB_algorithm` itself!"

    def giveFeedback(self,feedback):
        """Every PBMAB_algorithm needs a method `giveFeedback`. Note: Do NOT create an object of the superclass `PBMAB_algorithm` itself!"""
        assert False, "Every PBMAB_algorithm needs a method `giveFeedback`. Note: Do NOT create an object of the superclass `PBMAB_algorithm` itself!"

###############################################################################
# Class: PBMAB_Random (Subclass of PBMAB_algorithm)
###############################################################################
class PBMAB_Random(PBMAB_algorithm):
    """
This class models a PBMAB-algorithm, which chooses its queries uniformly at 
random from the set of all possible queries, i.e. of pairs [i,j] with 
0 <= i,j <= m-1 with i<j. (m: nr. of alternatives)
It ignores any feedback given to it.

EXAMPLE:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
np.random.seed(10)
m=5
A = PBMAB_Random(m)
TE=tenv.TestEnvironment(rr.sampleWST_boundedFromOneHalf(m,0.1,decimal_precision=3))
for i in range(0,123):
    [arm1,arm2] = A.getQuery()
    feedback = TE.pullArmPair(arm1,arm2)
    A.giveFeedback(feedback)
TE.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    def __init__(self,m):
        """m denotes the number of alternatives and has to be an integer >=2."""
        assert type(m) is int and m>=2, "'m' has to be an integer, which is >=2."
        self.m = m
        
    def getQuery(self):
        """
Returns the next query made by the algorithm. This query is sampled uniformly 
at random from the set of all possible queries.
m: positive integer
        """
        return(tenv.SampleRandomPair(self.m))
        
    def giveFeedback(self,feedback):
        """
'feedback' (which is ignored by the algorithm) has to be 'True' or 'false'. 
        """
        assert feedback==True or feedback==False, "The parameter `feedback` has to be either `True` or `False`."
        pass
    

###############################################################################
# Class: PBMAB_RoundRobin (Subclass of PBMAB_algorithm)
###############################################################################
class PBMAB_RoundRobin(PBMAB_algorithm):
    """
Models a PBMAB-algorithm, which chooses the queries in a RoundRobin manner, i.e.
the queries made are  -> ... -> [0,1] -> [0,2] -> ... -> [0,m-1] -> [1,2] -> ... -> [m-2,m-1] -> ...
It ignores any feedback given to it.

EXAMPLE:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
np.random.seed(10)
m=5
A = PBMAB_RoundRobin(m)
TE=tenv.TestEnvironment(rr.sampleWST_boundedFromOneHalf(m,0.1,decimal_precision=3))
for i in range(0,123):
    [arm1,arm2] = A.getQuery()
    feedback = TE.pullArmPair(arm1,arm2)
    A.giveFeedback(feedback)
TE.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    def __init__(self,m,startPair=[0,1]):
        """
Models a PBMAB-algorithm, which chooses the queries in a RoundRobin manner, i.e.
the queries made are  -> ... -> [0,1] -> [0,2] -> ... -> [0,m-1] -> [1,2] -> ... -> [m-2,m-1] -> ...
(m: nr. of alternatives).
The first query is 'startPair', then from this the RoundRobin procedure is started.  

m: integer, >=2.
startPair: list [i,j] of two distinct integers 0<=i,j<=m-1 with i<j.
        """
        assert type(m) is int and m>=2, "'m' has to be an integer, which is >=2."
        assert type(startPair)==list and len(startPair)==2 and type(startPair[0]) is int and type(startPair[1]) is int, "startPair has to be a list, which contains exactly two integers"
        assert 0<=startPair[0] and startPair[0]<startPair[1] and startPair[1]<=m-1
        self.m = m
        self.currentPair=startPair
    
    def getQuery(self):
        """
Outputs as query the current value of `self.currentPair` and updates `self.currentPair` afterwards according to RoundRobin,
i.e. the queries are -> ... -> [0,1] -> [0,2] -> ... -> [0,m-1] -> [1,2] -> ... -> [m-2,m-1] -> ...
        """
        response = self.currentPair.copy()
        if self.currentPair[1] >= self.m-1:
            if self.currentPair[0] < self.m -2: 
                self.currentPair[0] += 1
                self.currentPair[1] = self.currentPair[0]+1
            else:
                self.currentPair[0]=0
                self.currentPair[1]=1
        else:
            self.currentPair[1] += 1
        return(response)

    def giveFeedback(self,feedback):
        """
'feedback' (which is ignored by the algorithm) has to be 'True' or 'false'. 
        """
        assert feedback==True or feedback==False, "The parameter `feedback` has to be either `True` or `False`."
        pass

      


###############################################################################
# Class: PBMAB_RUCB (Subclass of PBMAB_algorithm)
###############################################################################
class PBMAB_RUCB(PBMAB_algorithm):
    """
EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
np.random.seed(10)
m=5
TE=tenv.TestEnvironment(rr.sampleWST_boundedFromOneHalf(m,0.1,decimal_precision=3))
A = PBMAB_RUCB(m,time = TE.time)    # We usually start with one comparison 
                                    # between every two arms, whence TE.time>0
print(TE.time)
for i in range(0,1000):
    [arm1,arm2] = A.getQuery()
    feedback = TE.pullArmPair(arm1,arm2)
    A.giveFeedback(feedback)
TE.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    def __init__(self,m,time=0):
        """
m: integer >=2
time: integer >=0 
        """
        assert type(m) is int and m>=2, "'m' has to be an integer, which is >=2."
        self.m = m
        self.B = []
        self.currentQuery = [0,1] #the pair, which should be compared next
        self.waitingForFeedback=False #Used for asserting that getQuery() and giveFeedback() is used alternating, started with getQuery()
        self.observedN = np.zeros((self.m,self.m))  #observedN[i,j] is the number of times the pair [i,j] has been observed until now. 
        self.observedR = np.zeros((self.m,self.m))  #observedR[i,j] is the number of times i has won agains j until now.
        self.time = time
        
    def getQuery(self):
        """
Returns the next query of the algorithm in form of a list [i,j], where i and j 
are distinct integers in 0,...m-1, namely the arms which should be compared 
next. Afterwards, feedback can be provided to the algorithm with the method 
"giveFeedback".

t = time step
n = (n)_ij number of comparisons between each arms i and j
w = (w)_ij number of wins of arm i against arm j
B = set of possible CWs (of last iteration?) ?? from RUCB paper...
        """
        assert self.waitingForFeedback is False, "Before doing the next query, use giveFeedback()."
        
        #Calculate the next query
        t = self.time
        n = self.observedN
        w = self.observedR
        B = self.B
        U = w/n + np.sqrt(2*math.log(t+1)/n)
        # x/0 := 1
        zeros = np.where(n == 0)
        for zero in range(len(zeros[0])):
            U[zeros[0][zero]][zeros[1][zero]] = 1
        # U_ii = 0.5
        for i in range(len(U)):
            U[i][i] = 0.5
        # find possible CWs
        C = []
        for arm in range(len(U)):
            if all(U[arm][i]>=0.5 for i in range(len(U[arm]))):
                C.append(arm)
        # if no possible CW exists sample random arm a
        if not C:
            arms = np.arange(len(U))
            a = random.choice(arms)
        if B:
            B = list(set(B).intersection(set(C)))
        # if one possible CW exists, chose it as arm a
        if len(C) == 1:
            B = C
            a = C[0]
        # if multiple possible CWs exist, chose arm a from it with specified probability
        if len(C) > 1:
            p = []
            for c in C:
                if c in B:
                    p.append(0.5)
                else:
                    p.append(1/math.pow(2,len(B))*len(list(set(C).difference(set(B)))))
            a = np.random.choice(C,p=p/np.sum(p))
        # chose second arm d as the one who has the highest probability to beat a
        D = np.argsort(U[:,a])
        # dont chose second = first arm
        if D[-1] == a:
            d = D[-2]
        else:
            d = D[-1]
        self.B = B
        self.waitingForFeedback = True
        self.currentQuery = [int(a),int(d)]
        return(self.currentQuery)
    
    
    def giveFeedback(self,feedback):    
        """
Returns "feedback" for the current query back to the algorithm. Before, 
"getQuery()" must have been executed. "feedback" has to be bool, and "True" 
means that currentQuery[0] wins against currentQuery[1] (Here: currentQuery=
getQuery() is the current query observed BEFORE executing "giveFeedback").
        """
        #Assertions:
        assert self.waitingForFeedback is True, "Use getQuery() before giving the feedback."
        assert feedback==True or feedback==False, "The parameter `feedback` (for `giveFeedback`) has to be a boolean." 
        
        #Update observedN and observedR accordingly
        self.observedN[self.currentQuery[0],self.currentQuery[1]] +=1
        self.observedN[self.currentQuery[1],self.currentQuery[0]] +=1
        if feedback==True:
            self.observedR[self.currentQuery[0],self.currentQuery[1]] +=1
        else:
            self.observedR[self.currentQuery[1],self.currentQuery[0]] +=1
        
        #Update the time
        self.time += 1
        self.waitingForFeedback = False


###############################################################################
# Class: PBMAB_DTS (Subclass of PBMAB_algorithm)
###############################################################################
class PBMAB_DTS(PBMAB_algorithm):
    """
EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
np.random.seed(11)
m=5
TE=tenv.TestEnvironment(rr.sampleWST_boundedFromOneHalf(m,0.1,decimal_precision=3))
A = PBMAB_DTS(m,time=TE.time)       # We usually start with one comparison 
                                    # between every two arms, whence TE.time>0
for i in range(0,1000):
    [arm1,arm2] = A.getQuery()
    feedback = TE.pullArmPair(arm1,arm2)
    A.giveFeedback(feedback)
TE.show()   
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    def __init__(self,m,time=0,N=[], R=[]):
        """
m: integer >=2
time: integer >=0
        """
        assert type(m) is int and m>=2, "'m' has to be an integer, which is >=2."
        assert N == [] or (type(N) is np.ndarray and N.shape[0] == m and N.shape[1] == m)
        assert R == [] or (type(R) is np.ndarray and R.shape[0] == m and R.shape[1] == m)
        self.m = m
        self.currentQuery = [0,1] #the pair, which should be compared next
        self.waitingForFeedback=False #Used for asserting that getQuery() and giveFeedback() is used alternating, started with getQuery()
        if N == []:
            self.observedN = np.zeros((self.m,self.m))  #observedN[i,j] is the number of times the pair [i,j] has been observed until now.
        else:
            self.observedN = N
        if R == []:
            self.observedR = np.zeros((self.m,self.m))  #observedR[i,j] is the number of times i has won agains j until now.
        else:
            self.observedR = R
        self.time = time
        
    def getQuery(self):
        """
Returns the next query of the algorithm in form of a list [i,j], where i and j 
are distinct integers in 0,...m-1, namely the arms which should be compared 
next. Afterwards, feedback can be provided to the algorithm with the method 
"giveFeedback".

t = time step
n = (n)_ij number of comparisons between each arms i and j
w = (w)_ij number of wins of arm i against arm j
        """
        assert self.waitingForFeedback is False, "Before doing the next query, use giveFeedback()."
        t = self.time
        n = self.observedN
        w = self.observedR
        
        # Upper Confidence Bound
        U = w/n + np.sqrt(2*math.log(t+1)/n)
        # Lower Confidence Bound
        L = w/n - np.sqrt(2*math.log(t+1)/n)
        # x/0 := 1
        zeros = np.where(n == 0)
        for zero in range(len(zeros[0])):
            U[zeros[0][zero]][zeros[1][zero]] = 1
            L[zeros[0][zero]][zeros[1][zero]] = 1
        # compute zeta: copeland score
        zeta = np.zeros(len(U))
        for i in range(len(U)):
            U[i][i] = 0.5
            L[i][i] = 0.5
            zeta[i] = len(np.where(U[i]>0.5)[0])/(len(U)-1)
        # chose copeland winner
        C = np.where(zeta == np.max(zeta))[0]
        # sample according to win probabilities
        theta_1 = np.zeros((len(U), len(U)))
        for i in range(len(U)):
            for j in range(len(U)):
                if i<j:
                    theta_1[i][j] = beta.rvs(w[i][j]+1,w[j][i]+1)
                    theta_1[j][i] = 1 - theta_1[i][j]
        # find copeland winner of sample
        copeland = np.array([len(np.where(theta_1[i]>0.5)[0]) for i in C])
        a = random.choice(np.where(copeland == np.max(copeland))[0])
        # sample according to win probabilities against arm a
        theta_2 = [beta.rvs(w[i][a]+1, w[a][i]+1) for i in range(len(U))]
        theta_2[a] = 0.5
        theta_2 = np.array(theta_2)
        # chose arm b with highest probability to beat a and low confidence (or unvisited)
        if theta_2[np.where((L[:, a] < 0.5) ^ (L[:, a] == 1))[0]].shape[0] == 0:
            B = np.arange(self.m)
            B = B[B!=a]
        else:
            B = np.where(theta_2 == np.max(theta_2[np.where((L[:, a] < 0.5) ^ (L[:, a] == 1))[0]]))[0]
        b = random.choice(B)
        self.currentQuery = [int(a),int(b)]
        #print(self.currentQuery)
        self.waitingForFeedback = True
        return(self.currentQuery)
        
    def giveFeedback(self,feedback):    
        """
Returns "feedback" for the current query back to the algorithm. Before, 
"getQuery()" must have been executed. "feedback" has to be bool, and "True" 
means that currentQuery[0] wins against currentQuery[1] (Here: currentQuery=
getQuery() is the current query observed BEFORE executing "giveFeedback").
        """
        #Assertions:
        assert self.waitingForFeedback is True, "Use getQuery() before giving the feedback."
        assert feedback==True or feedback==False, "The parameter `feedback` (for `giveFeedback`) has to be a boolean." 
        
        #Update observedN and observedR accordingly
        self.observedN[self.currentQuery[0],self.currentQuery[1]] +=1
        self.observedN[self.currentQuery[1],self.currentQuery[0]] +=1
        if feedback==True:
            self.observedR[self.currentQuery[0],self.currentQuery[1]] +=1
        else:
            self.observedR[self.currentQuery[1],self.currentQuery[0]] +=1
        
        #Update the time
        self.time += 1
        self.waitingForFeedback = False


# EXAMPLE CODE:
# print("Testing PBMAB_RUCB")
# np.random.seed(10)
# m=5
# TE=tenv.TestEnvironment(rr.sampleWST_boundedFromOneHalf(m,0.1,decimal_precision=3))
# A = PBMAB_RUCB(m,time = TE.time)    # We usually start with one comparison 
#                                     # between every two arms, whence TE.time>0
# print(TE.time)
# for i in range(0,1000):
#     [arm1,arm2] = A.getQuery()
#     feedback = TE.pullArmPair(arm1,arm2)
#     A.giveFeedback(feedback)
# TE.show()

# print("\n \n Testing PBMAB_DTS")
# np.random.seed(11)
# m=5
# TE=tenv.TestEnvironment(rr.sampleWST_boundedFromOneHalf(m,0.1,decimal_precision=3))
# A = PBMAB_DTS(m,time=TE.time)       # We usually start with one comparison 
#                                     # between every two arms, whence TE.time>0
# for i in range(0,1000):
#     [arm1,arm2] = A.getQuery()
#     feedback = TE.pullArmPair(arm1,arm2)
#     A.giveFeedback(feedback)
# TE.show()        


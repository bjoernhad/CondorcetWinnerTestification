#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This file implements the class 'CW_TestingComponent' and its superclass 
'TestingComponent'.
It is an implementation of Algorithm 1 (Noisy Tournament Sampling) in 
[Haddenhorst2021].

[Haddenhorst2021]: B. Haddenhorst, V. Bengs, J. Brandt and E.Hüllermeier,
Testification of Condorcet Winners in Dueling Bandits, Proceedings of UAI, 2021

"""
import TestEnvironment as tenv
import ReciprocalRelations as rr
import numpy as np
from scipy.special import binom
import math
import GraphTheoreticalPrerequisites as graph
import networkx as nx

###############################################################################
#   Class: TestingComponent
###############################################################################
class TestingComponent(object):
    def __init__(self):
        assert False, "Do NOT create an object of the superclass 'TestingComponent' itself!"
    
    def getDecision(self,N,R):
        assert False, "Every TestingComponent needs a method `getDecision`. Note: Do NOT create an object of the superclass `PBMAB_algorithm` itself!"


###############################################################################
#   Class: CW_TestingComponent (Subclass of TestingComponent)
###############################################################################

class CW_TestingComponent(TestingComponent):

    def __init__(self, N, R, h=0.01, gamma=0.1):
            """
            Initializes the LRT_TestingComponent instance
            N: NumPy array of size (m,m) (m: number of alternatives), where N[i,i]=0 and N[i,j]=N[j,i] for all i,j.
                (N[i,j] is the number of times i has been compared with j)
            R: NumPy array of size (m,m) with R[i,j]+R[j,i]=N[i,j] for all i,j.
                (R[i,j] is the number of times i has won against j)
            h: min difference of win probabilities to 0.5
            gamma: confidence in (0,1)
            """
            assert 0 < h < 0.5 and type(h) is float
            assert 0 < gamma < 1 and type(gamma) is float
            assert type(N) is np.ndarray and len(N.shape) == 2 and N.shape[0] == N.shape[
                1], "N either has to be `False` or a NumPy array of size (m,m) with N[i,i] = 0 and N[i,j]=N[j,i] for all distinct i,j. (Here, m is the number of alternatives)"
            self.m = N.shape[0]
            assert all(N[i, i] == 0 for i in range(0, self.m)) and (
                        (N - np.matrix.transpose(N)).astype(int) == np.zeros(
                    N.shape)).all(), "`N` has to fulfill N[i,i] = 0 and N[i,j]=N[j,i] for all distinct i,j."
            assert type(R) is np.ndarray and type(N) is np.ndarray and R.shape == (self.m, self.m) and (
                        R + np.matrix.transpose(
                    R) == N).all(), "`R` has to be `False` or a NumPy array of size (m,m) with R[i,j]+R[j,i]=N[i,j] for all i,j. ((Here, m is the number of alternatives)"
            self.N = N.astype(int)
            self.R = R.astype(int)
            self.time = int(np.sum(self.R))
            self.h = h
            self.gamma = gamma
            self.gamma_1 = self.gamma/self.m

    def update(self, arm1, arm2, feedback):
        """
        Updates the internal statistics after a comparison of 'arm1' with 'arm2' with outcome 'feedback'.
        (In case feedback==True, arm1 has won, otherwise arm2 has won.)
        arm1: integer in 0,1,...,m-1 (m:number of alternatives)
        arm2: integer in 0,1,...,m-1
        feedback: either True or False
        """
        assert type(arm1) is int and 0 <= arm1 and arm1 < self.m, "'arm1' has to be one of the integers 0,1,2,...,m-1."
        assert type(arm2) is int and 0 <= arm2 and arm2 < self.m, "'arm2' has to be one of the integers 0,1,2,...,m-1."
        assert feedback == True or feedback == False, "'feedback' has to be either True (meaning that arm1 won against arm2) or False (meaning that arm2 won against arm1"
        self.N[arm1, arm2] += 1
        if arm2 != arm1:
            self.N[arm2, arm1] += 1
        if feedback == True:
            self.R[arm1, arm2] += 1
        elif arm2 != arm1:
            self.R[arm2, arm1] += 1
        self.time += 1

    def C(self, N):
        """
        Compute Confidence interval boundary
        """
#        if N == 0:
#            return math.inf
#        eps = 0.5
#        delta = math.log(1 + eps) * math.pow(eps * self.gamma_1 / (2 + eps), 1 / (1 + eps))
#        return max(0, (1 + math.sqrt(eps)) * math.sqrt(
#            0.5 * (1 + eps) * math.log(math.log((1 + eps) * N) / delta) / N) - self.h)
        return (1/(2*N)) * np.ceil(np.log( (1-self.gamma_1) / self.gamma_1 ) / np.log( (0.5+self.h) / (0.5-self.h) )) 

    def TC(self):
        """
        Termination criteria which checks if every estimated win probability is confident enough
        """
        if np.sum(self.N) < self.m * self.m - self.m :
            # not all pairs are regarded once
            return False
        q = self.R / self.N
        # check if all estimated win probabilities are sure enough
        for line in range(self.m):
            for column in range(self.m):
                if line != column:
                    c = self.C(self.N[line, column])
                    if q[line, column] >= 0.5 - c and q[line, column] <= 0.5 + c:
                        return False
        # if termination criteria is fulfilled print the time
        print("Termination iteration: " + str(self.time))
        return True


    def DC(self):
        """
        Decision criteria which checks whether there exists a CW or not
        """
        q = self.R / self.N
        # check whether there exists an arm which have win probabilities > 0.5 against every other arm except itself
        for line in range(self.m):
            CW = True
            for column in range(self.m):
                if line != column:
                    if q[line, column] < 0.5:
                        # line is definitiv not the CW!
                        CW = False
                        continue
            if CW == True:
                # we have found a CW!
                break
        # printing the result
        if CW == True:
            print("CW exists!")
        else:
            print("CW does not exist!")
        return CW


###############################################################################
#   Class: Symmetric_TestingComponent (Subclass of TestingComponent)
###############################################################################

class Symmetric_TestingComponent(TestingComponent):

    def __init__(self, N, R, h=0.01, gamma=0.1):
            """
            Initializes the LRT_TestingComponent instance
            N: NumPy array of size (m,m) (m: number of alternatives), where N[i,i]=0 and N[i,j]=N[j,i] for all i,j.
                (N[i,j] is the number of times i has been compared with j)
            R: NumPy array of size (m,m) with R[i,j]+R[j,i]=N[i,j] for all i,j.
                (R[i,j] is the number of times i has won against j)
            h: min difference of win probabilities to 0.5
            gamma: confidence in (0,1)
            """
            #assert 0 < h < 0.5 and type(h) is float
            assert 0 < gamma < 1 and type(gamma) is float
            assert type(N) is np.ndarray and len(N.shape) == 2 and N.shape[0] == N.shape[
                1], "N either has to be `False` or a NumPy array of size (m,m) with N[i,i] = 0 and N[i,j]=N[j,i] for all distinct i,j. (Here, m is the number of alternatives)"
            self.m = N.shape[0]
            assert all(N[i, i] == 0 for i in range(0, self.m)) and (
                        (N - np.matrix.transpose(N)).astype(int) == np.zeros(
                    N.shape)).all(), "`N` has to fulfill N[i,i] = 0 and N[i,j]=N[j,i] for all distinct i,j."
            assert type(R) is np.ndarray and type(N) is np.ndarray and R.shape == (self.m, self.m) and (
                        R + np.matrix.transpose(
                    R) == N).all(), "`R` has to be `False` or a NumPy array of size (m,m) with R[i,j]+R[j,i]=N[i,j] for all i,j. ((Here, m is the number of alternatives)"
            self.N = N.astype(int)
            self.R = R.astype(int)
            self.time = int(np.sum(self.R))
            self.h = h
            self.gamma = gamma
            self.gamma_1 = self.gamma/self.m
            self.E = []
            self.G = nx.DiGraph()
            for i in range(0, self.m):
                self.G.add_node(i, key=i)

    def update(self, arm1, arm2, feedback):
        """
        Updates the internal statistics after a comparison of 'arm1' with 'arm2' with outcome 'feedback'.
        (In case feedback==True, arm1 has won, otherwise arm2 has won.)
        arm1: integer in 0,1,...,m-1 (m:number of alternatives)
        arm2: integer in 0,1,...,m-1
        feedback: either True or False
        """
        assert type(arm1) is int and 0 <= arm1 and arm1 < self.m, "'arm1' has to be one of the integers 0,1,2,...,m-1."
        assert type(arm2) is int and 0 <= arm2 and arm2 < self.m, "'arm2' has to be one of the integers 0,1,2,...,m-1."
        assert feedback == True or feedback == False, "'feedback' has to be either True (meaning that arm1 won against arm2) or False (meaning that arm2 won against arm1"
        self.N[arm1, arm2] += 1
        if arm2 != arm1:
            self.N[arm2, arm1] += 1
        if feedback == True:
            self.R[arm1, arm2] += 1
        elif arm2 != arm1:
            self.R[arm2, arm1] += 1
        self.time += 1
        self.update_edges(arm1, arm2)

    def update_edges(self, i, j):
        """
        checks whether the confidence for estimated win probabilities is high enough to add an edge
        i & j: arm pair which was compared
        """
        assert type(i) is int and i >= 0 and i <= self.m
        assert type(j) is int and j >= 0 and j <= self.m
        c = self.C(self.N[i, j])
        if self.R[i, j] / self.N[i, j] > 0.5 + c and not self.G.has_edge(j, i):
            self.G.add_edge(i, j)
        if self.R[i, j] / self.N[i, j] < 0.5 - c and not self.G.has_edge(i, j):
            self.G.add_edge(j, i)

    def C(self, N):
        """
        Compute Confidence interval boundary
        """
#        if N==0:
#            return math.inf
#        eps = 0.5
#        delta = math.log(1 + eps) * math.pow(eps * self.gamma / (2 + eps), 1 / (1 + eps))
#        return max(0, (1 + math.sqrt(eps)) * math.sqrt(
#            0.5 * (1 + eps) * math.log(math.log((1 + eps) * N) / delta) / N) - self.h)
        return (1/(2*N)) * np.ceil(np.log( (1-self.gamma_1) / self.gamma_1 ) / np.log( (0.5+self.h) / (0.5-self.h) )) 

    def TC(self):
        """
        checks whether every extension has a CW or has not a CW
        """
        if graph.is_CW_in_extension(self.G):
            # print result
#            print("Termination iteration: " + str(self.time))
            return True
        if graph.is_nonCW_in_extension(self.G):
            # print result
#            print("Termination iteration: " + str(self.time))
            return True
        return False

    def DC(self):
        """
        checks whether every extension has a CW or has not a CW
        """
        if graph.is_CW_in_extension(self.G):
            # print result
#            print("CW exists!")
            return True
        if graph.is_nonCW_in_extension(self.G):
            # print result
#            print("CW does not exist!")
            return False

    def find_CW(self):
        return graph.Find_CW(self.G)


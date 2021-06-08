#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This file contains the class TestEnvironment. An object of this class is an
environment for the dueling bandits scenario.
"""

import numpy as np
import ReciprocalRelations as rr

###############################################################################
# Function: SampleRandomPair
###############################################################################
def SampleRandomPair(m):
    """SampleRandomPair: Samples a pair [i,j] with 0<=i,j<=m-1 and i!=j, uniformly from the set of all such pairs
                        It returns a list.
            
            >> SampleRandomPair(10)
    """
    assert type(m) is int and m>=2, "'m' has to be an integer, which is >=2."
    sample=list()
    i = np.floor(np.random.uniform(0,m))
    j = np.floor(np.random.uniform(0,m-1))
    if j==i:
        j=j+1
    sample.append(int(i))
    sample.append(int(j))
    return(sample)


###############################################################################
# Class: TestEnvironment
###############################################################################
class TestEnvironment(object):
    def __init__(self,P,N=False,R=False):
        """TestEnvironment: Models the Dueling Bandits Setting. The ground-truth probabilities of alternative i winning against j is represented by a Reciprocal Relation P. 
            Moreover, it tracks the information N[i,j] how often alternatives i and j are compared 
            as well as the number R[i,j] how often alternative i won against j.
            The `time` is the total number of comparisons currently made, i.e. time = \sum_{i,j} R[i,j]"
            P: Reciprocal Relation, in which the (i,j)-entry P.getEntry([i,j]) denotes the probability that i is preferred to j   
            N: Either `False` or a NumPy array of size (m,m) with N[i,i]=0 and N[i,j]=N[j,i] for all i,j.
            R: Either `False` or a NumPy array of size (m,m) with R[i,j]+R[j,i]=N[i,j] for all i,j
            
            >> P=rr.sampleWST(3)
            >> N=np.array([[0,2,1],[2,0,1],[1,1,0]])
            >> R=np.array([[0,1,1],[1,0,0],[0,1,0]])
            >> TE = TestEnvironment(P,N,R)
        """
        assert type(P) is rr.ReciprocalRelation, "`P` has to be a Reciprocal Relation"
        assert (type(N)==bool and N==False) or (type(N) is np.ndarray and N.shape==(P.m,P.m)), "N either has to be `False` or a NumPy array of size (m,m) with N[i,i] = 0 and N[i,j]=N[j,i] for all distinct i,j. (Here, m is the number of alternatives, i.e. m=P.m)"
        assert (type(N)==bool and N==False) or (all(N[i,i]==0 for i in range(0,P.m)) and (N - np.matrix.transpose(N)==np.zeros(N.shape)).all()), "`N` has to fulfill N[i,i] = 0 and N[i,j]=N[j,i] for all distinct i,j."
        assert (type(R)==bool and R==False) or (type(R) is np.ndarray and type(N) is np.ndarray and R.shape==(P.m,P.m) and (R + np.matrix.transpose(R)==N).all()), "`R` has to be `False` or a NumPy array of size (m,m) with R[i,j]+R[j,i]=N[i,j] for all i,j. ((Here, m is the number of alternatives, i.e. m=P.m)"
        self.P=P            # The underlying ground-truth Reciprocal Relation.
        if type(N)==bool:
            self.N=np.array(np.zeros((P.m,P.m))+1,dtype=int)# number of times each pair is assumed to have been pulled at time 0
            for i in range(0,P.m): self.N[i,i]=0
        else:
            self.N=N.astype(int)           # N[i,j]: number of times, items i and j have been compared in total.
        
        if type(R)==bool:
            self.R=np.random.binomial(self.N,self.P.Q)     # every two arms get compared n times as initialization
            for i in range(0,self.P.m):                    # discard the samples for R[i,j],i>j, and enforce R[i,i]==0 and R[i,j]+R[j,i]==N[i,j] in the following
                self.R[i,i]
                for j in range(i+1,self.P.m):
                    self.R[j,i] = self.N[i,j]-self.R[i,j]
        else:
            self.R=R.astype(int)
        self.time= int(np.sum(self.R))  # number of comparisons which have been made up until now
        
    
    def show(self):
        """show: Method to show the internal statistics P,N,R and time. For Debugging.
            
            >> TE=TestEnvironment(rr.sampleWST(4))
            >> TE.show()
        """
        print("The (current) values of P,N,R and time are:\n",self.P.getRel(),",\n",self.N,",\n",self.R,",\n",self.time)
        
    def pullArmPair(self,i,j):
        """pullArmPair: Models one comparison between alternative i and alternative j
                    i: integer in 0,...,m-1 (m: number of alternatives)
                    j: integer in 0,...,m-1 (m: number of alternatives)
                    Returns "1" if i is the winner and "0" if j is the winner of the duel.
                    
                    >> TE=TestEnvironment(rr.sampleWST(4))
                    >> TE.pullArmPair(1,2)
        """
        assert type(i)==int and 0<=i and i<self.P.m, "`i` has to be an integer in 0,...,m-1 (m: number of alternatives)"
        assert type(j)==int and 0<=j and j<self.P.m,"`j` has to be an integer in 0,...,m-1 (m: number of alternatives)"
        assert i!=j, "i and j have to be two DISTINCT arms."
        self.N[i,j] += 1
        self.N[j,i] += 1
        # i wins against j with Prob. P[i,j] (Bernoulli-disitributed):
        winner = np.random.binomial(1,self.P.Q[i,j])  #winner=1 means i wins, winner=0 means j wins
        self.R[i,j] += winner 
        self.R[j,i] += 1-winner
        self.time += 1
        return(winner)
        
    def pullRandomArmPair(self):
        """pullRandomArmPair: Samples an arm uniformly at random from the set of all possible arms and  and pulls it. Returns the result in form of the lista list "[pair,winner] = [[pair[0],pair[1]],winner]".
    
                    >> TE=TestEnvironment(rr.sampleWST(4))
                    >> TE.pullRandomArmPair()
        """
        pair = SampleRandomPair(self.P.m)
        winner = self.pullArmPair(pair[0],pair[1])
        return([pair,winner])
    
    def pullAllArmPairs(self,number_of_times=1):
        """pullAllArmPairs: Pulls each pairs of arms `number_of_times` often
                            number_of_times: positive integer        
                    
                    >> TE=TestEnvironment(rr.sampleWST(4))
                    >> TE.pullAllArmPairs(3)
        """
        bufN = number_of_times * np.ones((self.P.m,self.P.m))
        for i in range(0,self.P.m):
            bufN[i,i] = 0
        bufR = np.random.binomial(bufN.astype(int),self.P.Q)
        for i in range(0,self.P.m):
            bufR[i,i] = 0
            for j in range(i+1,self.P.m):
                bufR[j,i] = number_of_times - bufR[i,j]
        self.__init__(self.P, (self.N + bufN).astype(int), (self.R + bufR).astype(int))
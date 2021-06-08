#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This file contains implementations of Deterministic Sequential Testing Algorithms
(DSTAs).
"""
import treelib as tlib
import numpy as np
import networkx as nx

###############################################################################
# Class: DeterministicTestingComponent
###############################################################################
class DeterministicTestingComponent(object):
    def __init__(self):
        assert False, "Do NOT create an object of the superclass 'TestingComponent' itself!"
        
    def getDecision(self,N,R):
        assert False, "Every TestingComponent needs a method `getDecision`. Note: Do NOT create an object of the superclass `PBMAB_algorithm` itself!"


###############################################################################
# Class: Optimal_Deterministic_CW_Tester(DeterministicTestingComponent)
###############################################################################
class Optimal_Deterministic_CW_Tester(DeterministicTestingComponent):
    """
This is an implementation of the optimal deterministic sequential testing 
algorithm for CW-testing of tournaments by {*} between the players 0,...,m-1.
It does this by determining a winner of KNOCKOUT-Tournament and takes this as 
the 'candidate' for the CW. Afterwards, it compares 'candidate' to all those 
players it has not been compared with before.
The internal value 'decision' is 'UNSURE' if the algo has not yet finished, and
it is 'not CW' or 'i' (for some i in 0,...,m) if it came to a decision yet.
Note that 'getQuery()' can only be executed when 'waitingForFeedback' is False
and similarly 'giveFeedback' only if 'waitingForFeedback' is True.

m: positive integer > 1

The theoretical worst-case sample complexity of this algorithm is given by
2*m-log2(m)-m. This is also demonstrated by the following EXAMPLE.

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
m = 7
Algo = Optimal_Deterministic_CW_Tester(m)
while Algo.decision == "UNSURE":
#    Algo.TT.T.show()
    [i,j] = Algo.getQuery()
    feedback = np.random.randint(2)
    Algo.giveFeedback(feedback)
#    print("Outcome of the duel ",i," versus ",j,":",feedback)
print("The decision of the algorithm is",Algo.decision)
print("The algorithm took ",Algo.nr_queries," queries for this, and 2m-log(m)-2 is",int(2*m-np.floor(np.log2(m))-2),".")
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

{*}: A. D. Procaccia. A note on the query complexity of the condorcet winner problem. Inf. Process.
Lett., 108:390–393, 2008.

    """
    def __init__(self,m):
        self.m = m
        self.TT = Tournament_Tree(m)    # For simulating the KNOCKOUT-Tournament
        self.compared_with = dict()     #self.compared_with[i] contains a list with the nodes which i has already been compared with
        for i in range(0,m):
            self.compared_with[i]=list()
        self.currentQuery = None        
        self.waitingForFeedback = False
        self.decision = "UNSURE"
        self.candidate = None       #is the winner of the tournament. If the tournament is not yet over, it's None.
        self.remaining_comparisons = list()
        self.nr_queries = 0
        self.G = nx.DiGraph()       #Collects all the duels and outcomes observed. Not necessary, only for Debugging purposes.
        for i in range(0, self.m):
            self.G.add_node(i, key=i)
        
    def getQuery(self):
        """
Returns the next query if testing is not over yet. Otherwise it returns False.
        """
        assert self.waitingForFeedback is False, "Before doing the next query, use giveFeedback()."
        self.waitingForFeedback = True
        if self.decision != "UNSURE":       # The Algorithm has already terminated -- It already came to a decision.
            return(False)
        if self.candidate != None:          #The tournament is already over, there is one remaining candidate
            self.currentQuery = self.remaining_comparisons.pop()
#            self.currentQuery = [self.candidate, opponent]
        else:                               # The tournament is not yet over.
            self.currentQuery = self.TT.allowed_duels[0]
        return(self.currentQuery)
    
    def giveFeedback(self,feedback):
        """
Allows to set the feedback for the duel between currentQuery[0] and 
currentQuery[1]. Here, 'True' means that currentQuery[0] has won.

feedback: boolean
        """
        assert self.waitingForFeedback is True, "Use getQuery() before giving the feedback."
        assert feedback==True or feedback==False, "The parameter `feedback` (for `giveFeedback`) has to be a boolean."
        [i,j] = self.currentQuery
        
        #Update G (only for debugging purposes)
        if feedback is True:
            self.G.add_edge(i,j)
        else:
            self.G.add_edge(j,i)
            
        self.compared_with[i].append(j)
        self.compared_with[j].append(i) 
        if self.candidate == None:      #The tournament is not yet over
            self.TT.conduct_duel(i,j,feedback)
            self.candidate = self.TT.winner
            if self.candidate != None:  # NOW, the tournament is over
                for k in range(0,self.m):
                    if k!= self.candidate and not (k in self.compared_with[self.candidate]):
                        self.remaining_comparisons.append([self.candidate,k])                
        else:   # The tournament has already been over
            if feedback == False and self.currentQuery[0]==self.candidate:           # self.candidate is NOT the CW
                self.decision = "Not CW"
            if feedback == True and len(self.remaining_comparisons)==0:     #self.candidate has won all of the remaining duels. Hence, it has to be the CW.
                self.decision = self.candidate
        self.waitingForFeedback = False
        self.nr_queries += 1
    
    def getCurrentDecision(self):
        return self.decision



###############################################################################
# Class: Tournament_Tree
###############################################################################
class Tournament_Tree():
    """
Models a tournament between the m players 0,...,m-1. The tournament is
represented as a binary tree T, which has the current candidates as leave
nodes -- The internal non-leave nodes are dummies and labeled with -1,-2,....
At each time, only such a duel can be conducted which is indeed a current duel
in the tournament, i.e., which is an element of 'allowed_duels'.
If a duel is conducted, the loser is removed from the tournament and the binary
tree T is updated accordingly.
If the tournament is over -- i.e., no more duels have to be made -- then T 
consists only of the winner node, and the variable 'winner' contains the
corresponding label (an element of 0,...,m-1). Otherwise, 'winner' is 'None'

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
TT = Tournament_Tree(5)
while TT.winner == None:
    TT.T.show()
    [i,j] = TT.allowed_duels[0]
    feedback= np.random.randint(2)
    TT.conduct_duel(i,j,feedback)
    print("Outcome of the duel ",i," versus ",j,":",feedback)
print("The WINNER of the tournament is:",TT.winner)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    def __init__(self,m):
        assert type(m) is int and m>=2, "m has t be an integer, which is at least 2."
        self.m = m
        self.T = self.__create_almost_complete_Tree__(m)
#        self.T.show()
        self.T = self.__shrink_Tree__(self.T)
#        self.T.show()
        self.allowed_duels = list()
        for i in range(0,m-1,2):
            self.allowed_duels.append([i,i+1])
        self.winner = None
        
    def __create_almost_complete_Tree__(self,m):
        """
Returns a Tree of type 'treelib.tree', which is almost complete, 
of height D=np.ceil(np.log2(m)) and contains exactly 0,...,m-1 as leave nodes.
The inner nodes are labeled by -2**D+1,...,-1 
m: a positive integer

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
T=create_almost_complete_Tree(13)
T.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        """
        assert type(m) is int and m>0, "'m' has to be a positive integer."
        T = tlib.Tree()
        D = int(np.ceil(np.log2(m)))
        nodes = list(range(m-1,-2**D,-1))
        root = nodes.pop()
        T.create_node(root,root,parent = None)
        parent_queue = [root]
        while len(nodes)>0:
            current_parent = parent_queue.pop()
            current_node = nodes.pop()
            T.create_node(current_node,current_node,parent = current_parent)
            parent_queue.insert(0,current_node)
            if len(nodes)>0:
                current_node = nodes.pop()
                T.create_node(current_node,current_node,parent = current_parent)
                parent_queue.insert(0,current_node)
        return(T)
    
    def __shrink_Tree__(self,T):
        """
Consecutively removes 
- all leave nodes, which are not labeled with 0,...,m-1
- all nodes, which have only one child.

T: Tree of type 'treelib.tree'.

For a concrete example see '__init__()'
        """
        finished = False
        while not finished:
            finished = True
            #Step 1: Remove all leave nodes, which are not labeled with 0,...m-1
            leaves = T.leaves()
            for node in leaves:
                if node.identifier < 0: 
                    T.remove_node(node.identifier)
                    finished = False
            #Step 2:  If a node v has only one children, delete it
            nodes = T.all_nodes()
            for node in nodes:
                if len(T.children(node.identifier))==1:
                    finished = False
                    child = T.children(node.identifier)[0]
                    if T.parent(node.identifier) == None:
                        T = T.subtree(child)
                    else:
                        T.link_past_node(node.identifier)
        return(T)
    
    
    def conduct_duel(self,i,j,feedback):
        """
Conducts a dual between i and j. If feedback is True, i is considered as the
winner, and otherwise j is considered to be the winner. The loser gets removed
from the tree. Moreover, if the "next opponent" X of the winner is determined, 
[winner,X] is added to 'allowed_duels'.

i,j: two current opponents in the tournament. Here, either [i,j] or [j,i] has 
    to be an element of 'allowed_duels'
feedback: boolean

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
TT = Tournament_Tree(5)
[i,j] = TT.allowed_duels[0]
TT.T.show()
print(str(i)," wins against ",str(j))
TT.conduct_duel(i,j,1)
TT.T.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        """
        assert feedback==True or feedback==False, "The parameter `feedback` (for `giveFeedback`) has to be a boolean."
        assert [i,j] in self.allowed_duels or [j,i] in self.allowed_duels, "There is currently no duel between i="+str(i)+" and j="+str(j)+" in the tournament-tree."
        
        if [i,j] in self.allowed_duels:
            self.allowed_duels.remove([i,j])
        else:
            self.allowed_duels.remove([j,i])
        
        if feedback == True:     #i is the winner
            winner, loser = i,j
        else:
            winner, loser = j,i
        self.T.remove_node(loser)
        parent = self.T.parent(winner)
        if self.T.parent(parent.identifier)==None:  # winner is the winner of the complete tournament
            self.T = tlib.Tree()
            self.T.create_node(winner,winner,parent=None)
            self.winner = winner
        elif parent != None and self.T.parent(parent.identifier)!=None: #The tournament is not yet over.
            self.T.link_past_node(parent.identifier) 
            sibling = self.T.siblings(winner)
            assert len(sibling)==1, "An error occured, "+str(winner)+" has siblings "+str(sibling)
            sibling = sibling[0].identifier
            if len(self.T.children(sibling))==0:
                self.allowed_duels.append([winner,sibling])
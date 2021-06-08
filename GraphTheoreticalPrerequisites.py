#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
from networkx.algorithms import tournament
from networkx.algorithms import dag
import matplotlib.pyplot as plt        #plt is required for some EXAMPLES
import ReciprocalRelations as rr
import numpy as np

is_tournament = tournament.is_tournament

################################################################################
## Function: reciprocalRelationToTournament
################################################################################
def reciprocalRelationToTournament(R):
    """
Tranforms a Reciprocal Relation R into a tournament G (of type 'networkx.DiGraph'),
where G cointans the edge (i,j) iff (R.Q[i,j]>0.5 or R[i,j]=0.5 and i<j).
R: ReciprocalRelation
Returns: tournament G (of type 'networkx.DiGraph')

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
R=rr.sampleReciprocal(5,decimal_precision=4)
R.show()
G=reciprocalRelationToTournament(R)
nx.draw_circular(G,with_labels=True)
plt.draw()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(R) is rr.ReciprocalRelation, "'R' has to be of type 'ReciprocalRelation'."
    m=R.m
    G = nx.DiGraph()
    for i in range(0,m):
        G.add_node(i,key = i)
        for j in range(i+1,m):
            G.add_edge(i,j) if R.Q[i,j]>=0.5 else G.add_edge(j,i)
    return(G)


################################################################################
## Function: TournamentToBinaryReciprocalRelation
################################################################################
def TournamentToBinaryReciprocalRelation(G):
    """
Transforms a Tournament G into a binary reciprocal relation R, where R.Q[i,j]=1
iff G has the edge (i,j).
G: tournament (of type 'networkx.DiGraph')
Returns: ReciprocalRelation (with entries 0,1)

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
R=rr.sampleReciprocal(5,decimal_precision=4)
R.show()
G=reciprocalRelationToTournament(R)
Rnew = TournamentToBinaryReciprocalRelation(G)
Rnew.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(G) is nx.DiGraph and tournament.is_tournament(G), "'G' has to be a tournament and of type 'networkx.DiGraph'."
    m = G.order()   #Number of nodes in the Graph
    I = np.zeros((m,m))
    for i in range(0,m):
        for j in range(i+1,m):
            I[i,j]=int(G.has_edge(i,j))
            I[j,i]=1-I[i,j]
    return(rr.ReciprocalRelation(I))


################################################################################
## Function: has_CW
################################################################################
def has_CW(G):
    """
Returns True if G has a Condorcet Winner, otherwise it returns False.
G: directed graph of type 'networkx.DiGraph'

Remark: This is a naive algorithm for checking for a CW, it is not optimal!

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
R = rr.getBinaryReciprocalRelation(4,7)
R.show()
G = reciprocalRelationToTournament(R)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("has CW?:",has_CW(G))
G.remove_edge(0,2)      # (0,2) is an edge in G and can thus be removed.
plt.figure(2)
nx.draw_circular(G,with_labels=True)
plt.show()
print("has CW?:",has_CW(G))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(G) is nx.DiGraph, "'G' has to be of type 'networkx.DiGraph'."
    nodes = list(G.nodes)
    for candidate in nodes:
        buf = True
        for node in nodes:
            if not G.has_edge(candidate,node) and not node==candidate:
                buf = False                     #candidate is NOT a CW
        if buf==True: 
            return(True)
    return(False)
    

################################################################################
## Function: Find_CW
################################################################################
def Find_CW(G):
    """
Returns the Condorcet Winner of G if it exists, otherwise it returns False.
G: directed graph of type 'networkx.DiGraph', it does NOT have to be a tournament!

Remark: This is a naive algorithm for finding the CW, it is not optimal!

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
R = rr.getBinaryReciprocalRelation(4,7)
R.show()
G = reciprocalRelationToTournament(R)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("Find_CW(G):",Find_CW(G))
G.remove_edge(0,2)      # (0,2) is an edge in G and can thus be removed.
plt.figure(2)
nx.draw_circular(G,with_labels=True)
plt.show()
print("Find_CW(G):",Find_CW(G))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """    
    assert type(G) is nx.DiGraph, "'G' has to be of type 'networkx.DiGraph'."
    nodes = list(G.nodes)
    for candidate in nodes:
        buf = True
        for node in nodes:
            if not G.has_edge(candidate,node) and not node==candidate:
                buf = False                     #candidate is NOT a CW
        if buf==True: 
            return(candidate)
    return(False)

################################################################################
## Function: Verify_i_as_CW
################################################################################
def Verify_i_as_CW(G,i):
    """
Returns True if i is the Condorcet Winner of G, otherwise it returns False.
G: directed graph of type 'networkx.DiGraph', does NOT have to be a tournament
i: a node in G

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
R = rr.getBinaryReciprocalRelation(4,7)
R.show()
G = reciprocalRelationToTournament(R)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("Verify_i_as_CW(G,0):",Verify_i_as_CW(G,0))
print("Verify_i_as_CW(G,1):",Verify_i_as_CW(G,1))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(G) is nx.DiGraph, "'G' has to be of type 'networkx.DiGraph'."
    nodes = list(G.nodes)
    assert i in nodes, "'i' as to be a node in G."
    buf = True
    for node in nodes:
        if not G.has_edge(i,node) and not node==i:
            buf = False                     # i is NOT a CW
    if buf==True: 
        return(True)
    return(False)

################################################################################
## Function: is_nonCW_in_extension
################################################################################    
def is_nonCW_in_extension(G):
    """
Returns True if G is 'non-CW in extension', otherwise it returns False.
G: directed graph of type 'networkx.DiGraph'

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
G=nx.DiGraph()
e_list = [(1,2),(0,1),(0,3),(3,1),(4,2),(4,0)]
G.add_edges_from(e_list)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("is_nonCW_in_extension(G):",is_nonCW_in_extension(G))
G.add_edge(3,4)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("is_nonCW_in_extension(G):",is_nonCW_in_extension(G))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(G) is nx.DiGraph, "'G' has to be of type 'networkx.DiGraph'."
    nodes = list(G.nodes)   
    for candidate in nodes:
        candidate_is_beaten = False
        for node in nodes:
            if G.has_edge(node,candidate):
                candidate_is_beaten = True
        if not candidate_is_beaten:
            return(False)           #Candidate can be the CW of an extension of G.
    return(True)


################################################################################
## Function: is_CW_in_extension
################################################################################
def is_CW_in_extension(G):
    """
Returns True if G is 'CW in expansion', otherwise it returns False.
G: directed graph of type 'networkx.DiGraph'

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
G=nx.DiGraph()
e_list = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)]
G.add_edges_from(e_list)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("is_CW_in_extension(G):",is_CW_in_extension(G))
G.remove_edge(0,1)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("is_CW_in_extension(G):",is_CW_in_extension(G))     
G.remove_edge(0,3)
plt.figure(1)
nx.draw_circular(G,with_labels=True)
plt.show()
print("is_CW_in_extension(G):",is_CW_in_extension(G))   
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(G) is nx.DiGraph, "'G' has to be of type 'networkx.DiGraph'."
    nodes = list(G.nodes)
    m = len(nodes)
    nr_beaten_list = np.zeros(m)    #nr_beaten_list[i] is the number of nodes v with an edge i->v in G if i is NOT beaten by any other node. Otherwise its -1
    for i in range(0,m):
        for j in range(0,m):
            if i != j and G.has_edge(nodes[i],nodes[j]) and nr_beaten_list[i] != -1:
                nr_beaten_list[i]+=1
            if i != j and G.has_edge(nodes[j],nodes[i]):
                nr_beaten_list[i]=-1
    #print(nr_beaten_list)
    if len(np.where(nr_beaten_list==m-1)[0]) >0:  #G has a CW
        return(True)
    buf = np.where(nr_beaten_list==m-2)[0]
    if len(buf)==2:
        [i0,i1] = buf
        if not G.has_edge(i0,i1) and not G.has_edge(i1,i0):     # There exist i0, i1 which are connected to every other node and i0 is not connected to i1
            return(True)
    return(False) 
    
################################################################################
## Function: is_beaten
################################################################################
def is_beaten(G,i):
    """
Returns True if G contains an edge (j,i)
G: directed graph of type 'networkx.DiGraph'
i: node in G

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
G=nx.DiGraph()
e_list = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)]
G.add_edges_from(e_list)
print(is_beaten(G,0),is_beaten(G,1))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(G) is nx.DiGraph, "'G' has to be of type 'networkx.DiGraph'."
    nodes = list(G.nodes)
    assert i in nodes
    for j in nodes:
        if G.has_edge(j,i):
            return(True)
    return(False)
    
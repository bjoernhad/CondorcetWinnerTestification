#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This file contains functions to deal with reciprocal relations.
    
    A reciprocal relation on [m] is represented as NumPy array of size (m,m),
    in particular the entries are indexed by [0,0],[0,1],...,[m-1,m-1].
    The diagonal entries are 0.5 by default, but as they are irrelevant, they are ignored.
    (E.g. they are not considered by the Function `isReciprocal`.)
    Non-diagonal entries have to be in the interval `[0,1]`.
"""

###############################################################################
# Required packages
###############################################################################
import numpy as np
import itertools as it #Required for samplePermutation
import scipy.special


###############################################################################
# Class: ReciprocalRelation
###############################################################################
class ReciprocalRelation(object):
    """
    ReciprocalRelation : Represents a reciprocal relation, which is internally saved as a NumPy array of size ((m,m)), which fulfills Q[j,i]=1-Q[i,j] at every time.
                            Here, the entries of Q are of type `float`, which are rounded up to `decimal_precision` in order to avoid errors (See {*} below).
                            The value `decimal_precision` is by default set to 10. 
                            
                            {*} With too high precision Python may calculate e.g. `1-0.9=0.09999999999999998 which leads to the error that it evaluates `1-0.9==0.1` as `False`
    """
    def __init__(self,Q=np.zeros((3,3)),decimal_precision=10):
        """ 
        __init__ : Creates a Reciprocal relation from a quadratic NumPy array Q. For this, ONLY the entries Q[i,j] with i<j are considered, the remaining are determined by Q[j,i]=1-Q[i,j] and Q[i,i]=0.5
                Q: quadratic NumPy array with entries in :math: `[0,1]`
                decimal_precision: The entries of the reciprocal relation are rounded up to `decimal_precision` in order to avoid errors (With too high precision Python may calculate e.g. `1-0.9=0.09999999999999998 which leads to the error that it evaluates `1-0.9==0.1` as `False`).
                
        """
        assert type(Q) is np.ndarray, "The parameter Q has to be a NumPy Array\n" #DEBUG
        assert len(Q.shape)==2 and Q.shape[0]==Q.shape[1], "The parameter Q has to be a QUADRATIC NumPy Array\n"#DEBUG
        assert type(decimal_precision) is int and 1<=decimal_precision, "`decimal_precision` has to be a positive integer"
        self.m = Q.shape[1]
        self.Q = Q
        self.precision = decimal_precision
        for i in range(0,self.m):
            self.Q[i,i]=0.5
            for j in list(range(i+1,self.m)):
                assert 0<=Q[i,j]<=1, "Non-diagonal entries of Q have to be in the interval :math:`[0,1]`." #DEBUG
                self.Q[i,j] = round(float(Q[i,j]),self.precision)
                self.Q[j,i] = round(float(1-Q[i,j]),self.precision)
    

    def getRel(self):
        """
        getRel: returns the reciprocal relation in form of a NumPy-Array
        """
        return self.Q 
    
    def setRel(self,Q):
        """
        set_Rel: Replaces the current Reciprocal Relation by ReciprocalRelation(Q). 
                In contrast to __init__,  Q has to be given here.
                Q: Quadratic NumPy-Array with entries in :math:`[0,1]`
        """
        self.__init__(Q)
    
    def getm(self):
        """getm : returns the value of `self.m`, i.e. the number of alternatives"""
        return self.m
    
    def setEntry(self,indices,value):
        """
        setEntry : Changes the entry at position `indices` to the new value `value`
                      indices: list of the form [i,j], where 0<=i,j<= m-1 are integers (m: number of alternatives)
                      value: a value in :math:`[0,1]`
        """
        assert type(indices)==list and len(indices)==2 and type(indices[0])==int==type(indices[1]), "`indices` has to be a list containing 2 integers" #DEBUG
        assert 0<=indices[0]<=self.m-1 and 0<=indices[1]<=self.m-1, "The elements in `indices` have to be >=0 and <=m-1" #DEBUG
        assert 0<=value<=1, "`value` has to fulfill 0<=value<=1"
        self.Q[indices[0],indices[1]]=round(float(value),self.precision)
        self.Q[indices[1],indices[0]]=round(float(1-value),self.precision)
    
    def getEntry(self,indices):
        """
        getEntry : Get the entry at position `indices`.
                      indices: list of the form [i,j], where 0<=i,j<= m-1 are integers (m: number of alternatives)
        """
        assert type(indices)==list and len(indices)==2 and type(indices[0])==int==type(indices[1]), "`indices` has to be a list containing 2 integers" #DEBUG
        assert 0<=indices[0]<=self.m-1 and 0<=indices[1]<=self.m-1, "The elements in `indices` have to be >=0 and <=m-1" #DEBUG
        return self.Q[indices[0],indices[1]]
    
    def addAlternative(self,values):
        """
        addAlternative : It adds a new alternative to the relation and with the values Q[i,m]=values[i] (and Q[m,i] = 1-values[i]), 1<=i<=m.
                            values: NumPy array with m elements, which are all numbers in `[0,1]` (m: prior number of alternatives)
                            
                            >> Q = ReciprocalRelation(np.zeros((2,2)))
                            >> Q.addAlternative(np.array([0.6,0.2]))
                            >> print(Q.getRel())
        """
        assert type(values)==np.ndarray and len(values)==self.m, "`values` has to be a NumPy array with m entries (m: current number of alternatives)" #DEBUG
        assert all(0<=elem<=1 for elem in values), "The entries of `values` have to be in :math:`[0,1]`" #DEBUG
        newQ = np.zeros((self.m+1,self.m+1))
        newQ[:self.m,:self.m] =self.Q[:self.m,:self.m]
        newQ[self.m,self.m] = 0.5
        for i in range(0,self.m):
            newQ[i,self.m] = round(float(values[i]),self.precision)
            newQ[self.m,i] = round(float(1-values[i]),self.precision)
        del self.Q    #Delete the old Q
        self.setRel(newQ)
    
    def RemoveAlternative(self,i):
        """
        RemoveAlternative : Removes the i-th alternative, i.e. the i-th row and i-th column, from the reciprocal relation.
                              i: Integer, in 0,...,m-1 (m : prior number of alternatives)
                              
                              >> Q=ReciprocalRelation(np.array([[0.5,0.3,0.4],[0.7,0.5,0.9],[0.6,0.1,0.5]]))
                              >> Q.RemoveAlternative(1)
                              >> print(Q.getRel())
                              [[0.5 0.4]
                              [0.6 0.5]]
        """
        assert type(i)==int and 0<=i<=self.m, "`i` has to be an integer in 0,...,m-1 (m : prior number of alternatives)"
        self.setRel(np.delete(np.delete(self.Q,obj=i,axis=0),obj=i,axis=1))
        
    def show(self,simplified=True):
        """
        show : Prints the Reciprocal Relation on the screen
                  If `Simplified`==True, only the entries above the diagonal are shown (as they determine the others)
                  If `Simplified`==False, all values are shown
                  
                  >> Q=ReciprocalRelation(np.array([[0.5,0.3,0.4],[0.7,0.5,0.9],[0.6,0.1,0.5]]))
                  >> Q.show()
                  |-      0.3     0.4     |
                  |-      -       0.9     |
                  |-      -       -       |
                  >> Q.show(False)
                  |0.5    0.3     0.4     |
                  |0.7    0.5     0.9     |
                  |0.6    0.1     0.5     |     
        """
        assert type(simplified)==bool, "`simplified` has to be a boolean" #DEBUG
        output = ""
        for i in range (0,self.m):
            output += "|"
            for j in range(0,self.m):
                buf = "- \t" if (j<=i and simplified==True) else str(self.Q[i,j])+" \t"
                output += buf
            output += "|\n" 
        print(output)
    
    def permute(self,perm):
        """
        permute : Permutes the reciprocal relation such that newQ[i,j]=oldQ[perm[i],perm[j]] for all i,j.
                    (Here, newQ/oldQ denote the reciprocal relation AFTER/BEfORE the permutation, resp.)
                    perm : list or NumPy array of length m, which contains all the elements 0,...,m-1 exactly once. (m : number of alternatives)
                    
                    >> Q=ReciprocalRelation(np.array([[0.5,0.3,0.4],[0.7,0.5,0.9],[0.6,0.1,0.5]]))
                    >> Q.show(False)
                    |0.5    0.3     0.4     |
                    |0.7    0.5     0.9     |
                    |0.6    0.1     0.5     |
                    >> Q.permute([1,2,0])
                    >> Q.show(False)
                    |0.5    0.9     0.7     |
                    |0.1    0.5     0.6     |
                    |0.3    0.4     0.5     |
        """
        assert (type(perm) is list or type(perm) is np.array) and all(type(elem)==int for elem in perm), "`perm` has to be a list of integers" #DEBUG
        assert len(perm)==self.m and all(elem in perm for elem in list(range(0,self.m))), "`perm` has to be a PERMUTATION, i.e. it has to contain all the elements 0,...,m-1 EXACTLY ONCE. (m : number of alternatives)" # DEBUG
        newQ = np.ones((self.m,self.m))
        for i in range(0,self.m):
            for j in range(i,self.m):
                newQ[i,j]=round(float(self.Q[perm[i],perm[j]]),self.precision)
                newQ[j,i]=round(float(self.Q[perm[j],perm[i]]),self.precision)
        del self.Q      #Delete the old Q
        self.setRel(newQ)
        
    def getDecimal_Precision(self):
        """getDecimal_Precision : Returns the value of `precision`
        """
        return self.precision
    
    def setDecimal_Precision(self,new_precision):
        """setDecimal_Precision : Sets the `decimal precision` to `new_precision`"""
        assert type(new_precision)==int and new_precision>=0, "`new_precision` has to be a non-negative integer"
        self.__init__(self.Q,new_precision)
    
    def isBinary(self):
        """
Returns 'True' if all entries of the relation are binary (0 or 1), otherwise it returns 'False'.
        """
        for i in range(0,self.m):
            for j in range(i+1,self.m):
                if self.Q[i,j] != 0 and self.Q[i,j] != 1:
                    return(False)
        return(True)
    
    def makeBinary(self):
        """
Changes all entries (Not on the diagonal) >=0.5 to '1' and the others to '0'.
        """
        for i in range(0,self.m):
            for j in range(i+1,self.m):
                if self.Q[i,j]>=0.5:
                    self.setEntry([i,j],1)
                else:
                    self.setEntry([i,j],0)
        return(True)
    
    def copy(self):
        """
Returns a deep copy of this ReciprocalRelation. 

EXAMPLE
>> Q=ReciprocalRelation(np.array([[0.5,0.3,0.4],[0.7,0.5,0.9],[0.6,0.1,0.5]]))
>> newQ = Q.copy()
>> Q.setEntry([0,1],0.99)
>> Q.show()
>> newQ.show()
"""
        return(ReciprocalRelation(self.Q.copy(),self.precision))

###############################################################################
# Function: sampleReciprocal
###############################################################################
def sampleReciprocal(m,decimal_precision=10):
    """sampleReciprocal : Returns a Sample of a Reciprocal Relation with m alternatives, where each entry above the diagonal is uniformly sampled from :math:`[0,1]`
            m: positive integer
            decimal_precision: decimal_precision of the Reciprocal Relation
            
            Example:
            >> Q = sampleReciprocal(3,decimal_precision=3)
            >> Q.show()
            |-      0.217   0.329   |
            |-      -       0.096   |
            |-      -       -       |
    """
    assert type(m) is int and m>=1, "The parameter `m` has to be a positive integer." #DEBUG
    A=np.zeros((m,m))
    Q=np.random.uniform(A+0,A+1)
    return(ReciprocalRelation(Q,decimal_precision))
    


###############################################################################
# Function: isWST
###############################################################################
def isWST(Q,proof=False):
    """isWST: Checks whether a Reciprocal Relation Q is WST. (If Q is not a Reciprocal Relation, it throws an `AssertionError`.)
    If `proof` is `True` and Q is not WST, it returns the `False` and the triplet (i,j,k) where the violation occurs, 
    i.e. Q[i,j]>=0.5, Q[j,k]>=0.5 and Q[i,k]<0.5 holds.
    
    >> Q=ReciprocalRelation(np.array([[0.5,0.3,0.4],[0.7,0.5,0.9],[0.6,0.1,0.5]]))
    >> print(isWST(Q))
    True
    >> Q=ReciprocalRelation(np.array([[0.5,0.3,0.6],[0.7,0.5,0.1],[0.4,0.9,0.5]]))
    >> print(isWST(Q,proof=True))
    (False, (0, 2, 1))
    """
    assert type(Q)==ReciprocalRelation, "The parameter Q has to be a ReciprocalRelation" #DEBUG
    Q=Q.getRel()
    m=Q.shape[1]
    for i in list(range(0,m)):
        for j in list(range(0,m)):
            for k in list(range(0,m)):
                if i!=j and i!=k and j!=k:
                    if Q[i,j]>=0.5 and Q[j,k]>=0.5 and Q[i,k]<0.5:
                        return(False if proof==False else (False, (i,j,k)))
    return(True)
    

###############################################################################
# Function: samplePermutation
###############################################################################
def samplePermutation(m):
    """samplePermutation : Returns a Sample permutation on [m], returns an object of the class :class:`list`
    
    >> print(samplePermutation(5))
    [3, 4, 1, 2, 0]
    """
    assert type(m)==int, "The parameter `m` has to be of type `int`."
    x = list(range(0,m))
    np.random.shuffle(x)
    return(x)


###############################################################################
# Function: sampleWST
###############################################################################
def sampleWST(m,decimal_precision=10):
    """sampleWST: Returns a Sample of a Reciprocal Relation with `m` alternatives, which is WST.
                  Sampling is done by at first choosing the entries above the diagonal uniformly in [0.5,1] and then randomly permuting rows and columns.)
                  m: Number of Alternatives
                  decimal_precision: decimal_precision of the resulting Reciprocal Relation
                  
                  >> Q = sampleWST(4,decimal_precision=3)
                  >> Q.show()
                  |-      0.195   0.797   0.967   |
                  |-      -       0.625   0.795   |
                  |-      -       -       0.965   |
                  |-      -       -       -       |
    """
    assert type(m) is int and m>=1, "The parameter `m` has to be a positive integer." #DEBUG
    A=np.zeros((m,m))
    Q=ReciprocalRelation(np.random.uniform(A+0.5,A+1),decimal_precision)
    while isWST(Q)==False:      #In case Q contains the entry `0.5`, Q may not be WST.  This is VERY improbable. If so, resample.
        Q=ReciprocalRelation(np.random.uniform(A+0.5,A+1),decimal_precision) 
    Q.permute(samplePermutation(m))
    return(Q)


###############################################################################
# Function: sampleWST_boundedFromOneHalf
###############################################################################
def sampleWST_boundedFromOneHalf(m,h,decimal_precision=10):
    """sampleWST_boundedFromOneHalf: Returns a Sample of a Reciprocal Relation Q with `m` alternatives, which is WST and whose entries Q[i,j] fulfill |Q[i,j]-0.5|>h.
                    m: Number of Alternatives
                    h: `float` in the interval :math:`(0,0.5)`
                    decimal_precision: decimal_precision of the resulting Reciprocal Relation
                    
                    >> Q = sampleWST_boundedFromOneHalf(3,0.2,4)
                    >> Q.show()
                    |-      0.9355  0.7037  |
                    |-      -       0.8775  |
                    |-      -       -       |
                    
                  Returns a Sample n times n relation Q which is WST with values |Q[i,j]-1/2|>=h
    """
    assert type(m) is int and m>=1, "The parameter `m` has to be a positive integer." #DEBUG
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    A=np.zeros((m,m))
    Q=ReciprocalRelation(np.random.uniform(A+0.5+h,A+1),decimal_precision)
    assert isWST(Q), "An error occured, Q sampled in `sampleWST_boundedFromOneHalf` is not WST." #DEBUG
    Q.permute(samplePermutation(m))
    return(Q)
 

    
###############################################################################
# Function: sampleNotWST
###############################################################################
def sampleNotWST(m,max_tries=1000,decimal_precision=10):
    """sampleNotWST: Returns a Sample of a Reciprocal Relation Q with `m` alternatives, which is not WST.
                    (It samples in an acception-rejection manner consecutively ReciprocalRelations and terminates as soon as the first one is NotWST.)
                    m: Number of Alternatives
                    max_tries: maximum number of tries for the acception-rejection procedure, i.e. maximum number of Reciprocal Relations sampled
                    decimal_precision: decimal_precision of the resulting Reciprocal Relation
                    
                    >> Q = sampleNotWST(4,100,3)
                    >> Q.show()
                    |-      0.9     0.795   0.163   |
                    |-      -       0.673   0.786   |
                    |-      -       -       0.489   |
                    |-      -       -       -       |
    """
    assert type(m) is int and m>= 1, "`m` has to be a positive integer."
    assert type(max_tries) is int and m>=1, "`max_tries` has to be a positive integer."
    assert type(decimal_precision) is int and m>=1, "`decimal_precision` has to be a positive integer."                
    Q=sampleReciprocal(m,decimal_precision)
    counter=0
    while isWST(Q) and counter < max_tries:
        Q=sampleReciprocal(m,decimal_precision)
        counter += 1
    assert not isWST(Q), "ERROR, ALL of the "+str(max_tries)+ " reciprocal samples tested were WST"
    return(Q)
    
###############################################################################
# Function: sampleNotWST_boundedFromOneHalf
###############################################################################
def sampleNotWST_boundedFromOneHalf(m,h,max_tries=1000,decimal_precision=10):
    """sampleNotWST_boundedFromOneHalf : Returns a Sample of a Reciprocal Relation Q with `m` alternatives, which is not WST and with all entries Q[i,j] such that |Q[i,j]-0.5|>0.5.
            At first, `SampleNotWST` is used, then `|Q[i,j]-0.5|>h for all i,j` is enforced by `__EnforcedBoundedFromOneHalf__`.
            m: Number of Alternatives
            h: `float` in the interval :math:`(0,0.5)` 
            max_tries: maximum number of tries for the acception-rejection procedure, i.e. maximum number of Reciprocal Relations sampled
            decimal_precision: decimal_precision of the resulting Reciprocal Relation
            
            >> Q = sampleNotWST_boundedFromOneHalf(3,0.3,100,3)
            >> Q.show()
            |-      0.0092  0.9876  |
            |-      -       0.1828  |
            |-      -       -       |
    """
    assert type(m) is int and m>= 1, "`m` has to be a positive integer."
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    assert type(max_tries) is int and m>=1, "`max_tries` has to be a positive integer."
    assert type(decimal_precision) is int and m>=1, "`decimal_precision` has to be a positive integer."  
    Q=sampleNotWST(m,max_tries,decimal_precision)
    Q=__EnforceBoundedFromOneHalf__(Q,h)
    return(Q)
    
################################################################################
## Function: __EnforceBoundedFromOneHalf__
################################################################################
def __EnforceBoundedFromOneHalf__(Q,h):
    """__EnforceBoundedFromOneHalf__ : Given a reciprocal relation Q, it constructs a relation Q', whose values fulfill |Q'[i,j]-0.5|>=h.
            For this, it changes all entries Q[i,j]>=0.5 to 2*(0.5-h)*Q[i,j]+2*h and all entries Q[i,j]<0.5 to Q[i,j]=2*(0.5-h)*Q[i,j].
            Q: ReciprocalRelation
            h: `float` in the interval :math:`(0,0.5)` 
            
            >> Q = sampleReciprocal(3,3)
            >> Q.show()
            |-      0.896   0.732   |
            |-      -       0.231   |
            |-      -       -       |
            >> Q=__EnforceBoundedFromOneHalf__(Q,0.4)
            >> Q.show()
            |-      0.9792  0.9464  |
            |-      -       0.0462  |
            |-      -       -       |
    """
    assert type(Q) is ReciprocalRelation, "`Q` has to be a ReciprocalRelation."
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    Q=Q.Q
    m=Q.shape[1]
    for i in range(0,m):
        for j in range(i+1,m):
            if Q[i,j]>=1/2: 
                Q[i,j] = 2*(1/2-h)*Q[i,j] + 2*h
            else:
                Q[i,j] = 2*(1/2-h)*Q[i,j]
    return(ReciprocalRelation(Q))


###############################################################################
# Function: getGTperm
###############################################################################
def getGTperm(Q):
    """getGTperm: If the Reciprocal Relation Q is WST, it returns the ground-truth underlying permutation `perm`, i.e. Q[perm[i],perm[j]]>=0.5 for all i<j.
                  If Q is not WST, `False` is returned.
                  Q: Reciprocal Relation 
                  
                  >> Q=sampleNotWST(4,100,3)
                  >> print(getGTperm(Q))
                  False
                  >> Q=sampleWST(4,3)
                  >> Q.show()
                  |-      0.396   0.782   0.3     |
                  |-      -       0.787   0.544   |
                  |-      -       -       0.306   |
                  |-      -       -       -       |
                  >> print(getGTperm(Q))
                  [1 3 0 2]
    """    
    assert type(Q)==ReciprocalRelation, "`Q` has to be a ReciprocalRelation."
    if not isWST(Q): 
        return(False)
    Q=Q.Q
    m=Q.shape[1]                
    perms=list(it.permutations(range(0,m)))
    for k in range(0,len(perms)):
        perm = perms[k]
        perm_is_GT=1
        for i in range(0,m):
            for j in range(i+1,m):
                if Q[perm[i],perm[j]]<0.5:
                    perm_is_GT = 0
        if perm_is_GT == 1:
            return(np.array(perm))
    assert False, "An error occured, Q is WST but I could not find the underlying ground-truth permutation."


###############################################################################
# Function: getBinaryWSTfromPerm
###############################################################################
def getBinaryWSTfromPerm(perm):
    """getBinaryWSTfromPerm: returns the binary reciprocal relation Q with Q[perm[i],perm[j]]=1 iff i<j 
            perm : list or NumPy array of length m, which contains all the elements 0,...,m-1 exactly once. (m : number of alternatives)
            
            >> Q=getBinaryWSTfromPerm([3,1,2,0])
            >> Q.show()
            |-      0.0     0.0     0.0     |
            |-      -       1.0     0.0     |
            |-      -       -       0.0     |
            |-      -       -       -       |
            >> print(getGTperm(Q))
            [3 1 2 0]    
    """
    assert (type(perm) is list or type(perm) is np.array) and all(type(elem)==int for elem in perm), "`perm` has to be a list of integers" #DEBUG
    assert all(elem in perm for elem in list(range(0,len(perm)))), "`perm` has to be a PERMUTATION, i.e. it has to contain all the elements 0,...,m-1 EXACTLY ONCE. (m : number of alternatives)" # DEBUG
    m=len(perm)
    Q=np.zeros((m,m))
    for i in range(0,m-1):
        for j in range(i+1,m):
            Q[perm[i],perm[j]]=1
    return(ReciprocalRelation(Q))


###############################################################################
# Function: getCopelandRanking
###############################################################################
def getCopelandRanking(Q,print_scores=False):
    """getCopelandRanking: Returns the corresponding Copeland ranking `perm` of the Reciprocal relation `Q` as a NumPy array. Here,  ties are broken uniformly at random. 
            That is, if s[perm[0]] >= s[perm[1]] >= s[perm[2]] >= ... >= s[perm[m-1]] holds, where s[j] is the Copeland score of alternative j: 
            (That is, s[j] = (number of alternatives j' with Q[j,j']>0.5) + \sum_{j' with Q[j,j']=0.5} X_{j,j'}, where X_{j,j'} =Bernoulli(0.5))
            Note: If Q is WST, then this is a the corresponding WST ranking.
            Q: ReciprocalRelation
            print_scores: bool. If `print_scores` is True, the vector s Containing the Copeland Scores is printed. (For Debugging)
            
>> Q=ReciprocalRelation(np.array([[0.5,0.7,0.4,0.3],[0.3,0.5,0.9,0.8],[0.6,0.1,0.5,0.2],[0.4,0.9,0.6,0.7]]))
>> print(getCopelandRanking(Q,print_scores=True))
The Copeland-Scores for Q are: [1. 2. 1. 2.]
[1 3 2 0]
            
            # As ties are broken at random,  the outcome of `print(getCopelandRanking(Q))` may differ from time to time.
    """
    assert type(Q) is ReciprocalRelation, "`Q` has to be a Reciprocal Relation."
    assert type(print_scores) is bool, "`print_scores` has to be a boolean."
    Q = Q.Q
    m=Q.shape[1]
    s=np.zeros(m)   #In `s` the Copeland scores are saved.
    
    # Step 1: Calculate the Copeland Scores
    for i in range(0,m):
        for j in range(i+1,m):
            if Q[i,j]>0.5:
                s[i]=s[i]+1
            elif Q[i,j]<0.5:
                s[j]=s[j]+1
            else:   #Q[i,j]=0.5
                b=int(np.floor(np.random.uniform(0,2)))
                if b==1:
                    s[i]=s[i]+1
                else:
                    s[j]=s[j]+1
    if print_scores is True: print("The Copeland-Scores for Q are: "+str(s)) #DEBUG
                
    # Step 2: Break ties uniformly at random
    for l in range(0,m):
        positions = np.where(s==l)[0] 
        number = len(positions)
        if number >=2:
            modifier = np.array(range(0,number))/number
            np.random.shuffle(modifier)
            s[positions]+=modifier
    return(np.argsort(s)[::-1])    #Return the descending argsort of s.

###############################################################################
# Function: getBinaryRelation
###############################################################################
def getBinaryReciprocalRelation(m,idx):
    """getBinaryReciprocalRelation: Returns for integer idx between 0 and 2^\binom{m}{2}-1 the corresponding binary relation associated with x, see {^}
            m: positive integer
            idx: integer with 0 <= idx < 2**(scipy.special.binom(m,2))
            
{^}: Given x, we at first calculate the binary representation of x (eg. "110" in case x=6). Then, we run through this representation (from RIGHT to LEFT) and fill the 
a reciprocal relation R (traversed in RoundRobin-manner started at [0,1], i.e. [0,1]->[0,2]->...->[0,m]->[1,2]->...->[m-1,m]) and place the entries
of the binary representation of "x" therein (e.g. in case x=6 we choose R.Q[0,1]=1, R.Q[0,2]=1, R.Q[1,2]=0)
    """
    assert type(m) is int and m>=1, "`m` has to be a positive integer."
    assert type(idx) is int and 0<=idx and idx < 2**(scipy.special.binom(m,2)),"There does not exist a binary relation correlated to x"
    b=bin(int(idx))
    position=len(b)-1
    Q=np.zeros((m,m))
    for i in list(range(0,m)):
        for j in list(range(i+1,m)):
            #print(i,j)
            if position>=2:
                Q[i,j]=b[position]
                position-=1
            else:
                Q[i,j]=0
    return(ReciprocalRelation(Q))


###############################################################################
# Function: getAllWSTIndices
###############################################################################
def getAllWSTIndices(m):
    """
Returns a list of exactly those integers x in 0,1,...,2**binom(m,2)-1, for 
which 'getBinaryReciprocalRelation(m,x)' is a WST reciprocal relation.
m: positive integer 
Returns: list
    """
    assert type(m) is int and m>=1, "`m` has to be a positive integer."
    AllWSTIndices = list()
    for idx in range(0,2**(scipy.special.binom(m,2)).astype(int)):
        if isWST(getBinaryReciprocalRelation(m,idx)) is True:
            AllWSTIndices.append(idx)
    return(AllWSTIndices)

###############################################################################
# Function: getIndexOfBinaryRelation
###############################################################################
def getIndexOfBinaryRelation(R):
    """
Returns the corresponding index 'idx' of a binary reciprocal relation R, such
that getBinaryReciprocalRelation(R.m,idx) and R have the same entries everywhere
R: Reciprocal Relation, where all entries are in {0,1}

The following EXAMPLE demonstrates that the code is correct
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for idx in range(0,64):     #note that 2**binom(4,2)=64
    R = getBinaryReciprocalRelation(4,idx)
    assert getIndexOfBinaryRelation(R)==idx, "ERROR"
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Note: The code shows, that this function is a left-inverse of the function
getBinaryReciprocalRelation(). As (for fixed m) the domain and co-domain are both
finite, this must also be a right-inverse.
"""
    assert type(R) is ReciprocalRelation and R.isBinary(), "'R' has to be a binary relation"
    index_str = ""
    for i in range(0,R.m):
        for j in range(i+1,R.m):
            index_str = str(int(R.Q[i,j])) + index_str
    return(int(index_str, base=2))
    

###############################################################################
# Function: has_CW
###############################################################################
def has_CW(Q):
    """
Returns True if Q has a Condorcet winner, otherwise it returns False
Q: ReciprocalRelation

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = getBinaryReciprocalRelation(4,0)
Q.show(False)
print(has_CW(Q))
Q2 = getBinaryReciprocalRelation(4,6)
Q2.show(False)
print(has_CW(Q2))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(Q) is ReciprocalRelation
    for i in range(0,Q.m):
        i_is_CW = True
        for j in range(0,Q.m):
            if i != j and Q.Q[i,j]<0.5:
                i_is_CW = False
        if i_is_CW is True:
            return(True)
    return(False)

###############################################################################
# Function: get_CW
###############################################################################
def get_CW(Q):
    """
Returns i if i is the CW of Q. If Q has no CW, it returns False
Q: ReciprocalRelation

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = getBinaryReciprocalRelation(4,0)
Q.show(False)
print(get_CW(Q))
Q2 = getBinaryReciprocalRelation(4,6)
Q2.show(False)
print(get_CW(Q2))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(Q) is ReciprocalRelation
    for i in range(0,Q.m):
        i_is_CW = True
        for j in range(0,Q.m):
            if i != j and Q.Q[i,j]<0.5:
                i_is_CW = False
        if i_is_CW is True:
            return(i)
    return(False)


###############################################################################
#   In the following, we provide different functions to uniformly sample a 
#   reciprocal relation from a set Q_m^'. Here is an overview:
#   
#   sampleCW                            Q_{m}(CW)
#   sampleCW_boundedFromOneHalf         Q_{m}^{h}(CW)
#   sampleNotCW                         Q_{m}(\neg CW)
#   sampleNotCW_boundedFromOneHalf      Q_{m}^{h}(\neg CW)
#   sampleRecRel_exactly_h              \hat{Q}_{m}^{h}
#   sampleCW_exactly_h                  \hat{Q}_{m}^{h}(CW)
#   sampleNotCW_exactly_h               \hat{Q}_{m}^{h}(\neg CW)
###############################################################################

    
###############################################################################
# Function: sampleCW
###############################################################################
def sampleCW(m,decimal_precision=10):
    """
Samples uniformly at random a reciprocal relation Q with m alternatives, which 
has a CW

m: number of arms
decimal_precision: parameter for ReciprocalRelation

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = sampleReciprocal(5,2)
Q.show()
print(has_CW(Q))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    Q = sampleReciprocal(m,decimal_precision)        
    cw = np.random.randint(0,m)       # cw is chosen to be the CW
    for j in range(0,m):
        if Q.Q[cw,j]<0.5:
            buf = Q.Q[j,cw]
            Q.setEntry([cw,j],buf)
    return(Q), cw
    

###############################################################################
# Function: sampleCW_boundedFromOneHalf
###############################################################################
def sampleCW_boundedFromOneHalf(m,h,decimal_precision=10):
    """sampleCW_boundedFromOneHalf: Returns a Sample of a Reciprocal Relation Q with `m` alternatives, which is CW and whose entries Q[i,j] fulfill |Q[i,j]-0.5|>=h.
                    m: Number of Alternatives
                    h: `float` in the interval :math:`(0,0.5)`
                    decimal_precision: decimal_precision of the resulting Reciprocal Relation

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = sampleCW_boundedFromOneHalf(5,0.1,2)
Q.show()
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(m) is int and m>=1, "The parameter `m` has to be a positive integer." #DEBUG
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    Q, cw = sampleCW(m,decimal_precision)
    Q =  __EnforceBoundedFromOneHalf__(Q,h)
    Q = ReciprocalRelation(Q.Q,decimal_precision)
    #assert has_CW(Q), "An error occured"
    return(Q, cw)


###############################################################################
# Function: sampleNotCW
###############################################################################
def sampleNotCW(m,max_tries=1000,decimal_precision=10):
    """sampleNotCW: Returns a Sample of a Reciprocal Relation Q with `m` alternatives, which is not CW.
                    (It samples in an acception-rejection manner consecutively ReciprocalRelations and terminates as soon as the first one is NotCW.)
                    m: Number of Alternatives
                    max_tries: maximum number of tries for the acception-rejection procedure, i.e. maximum number of Reciprocal Relations sampled
                    decimal_precision: decimal_precision of the resulting Reciprocal Relation
                    
EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = sampleNotCW(4,max_tries=1000,decimal_precision=2)
Q.show(False)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(m) is int and m>= 1, "`m` has to be a positive integer."
    assert type(max_tries) is int and m>=1, "`max_tries` has to be a positive integer."
    assert type(decimal_precision) is int and m>=1, "`decimal_precision` has to be a positive integer."                
    Q=sampleReciprocal(m,decimal_precision)
    counter=0
    while has_CW(Q) and counter < max_tries:
        Q=sampleReciprocal(m,decimal_precision)
        counter += 1
    assert not has_CW(Q), "ERROR, ALL of the "+str(max_tries)+ " reciprocal samples tested were WST"
    return(Q)


###############################################################################
# Function: sampleCW_boundedFromOneHalf
###############################################################################
def sampleNotCW_boundedFromOneHalf(m,h,max_tries=1000,decimal_precision=10):
    """sampleNotCW_boundedFromOneHalf: Returns a Sample of a Reciprocal Relation Q with `m` alternatives, which is Not CW and whose entries Q[i,j] fulfill |Q[i,j]-0.5|>=h.
                    m: Number of Alternatives
                    h: `float` in the interval :math:`(0,0.5)`
                   max_tries: maximum number of tries for the acception-rejection procedure, i.e. maximum number of Reciprocal Relations sampled
                    decimal_precision: decimal_precision of the resulting Reciprocal Relation

EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = sampleNotCW_boundedFromOneHalf(5,0.4,max_tries=1000,decimal_precision=2)
Q.show(False)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    assert type(m) is int and m>=1, "The parameter `m` has to be a positive integer." #DEBUG
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    Q = sampleNotCW(m, max_tries, decimal_precision)
    Q =  __EnforceBoundedFromOneHalf__(Q,h)
    Q = ReciprocalRelation(Q.Q,decimal_precision)
    assert not has_CW(Q), "An error occured"
    return(Q)



###############################################################################
#   Function: sampleRecRel_exactly_h
###############################################################################
def sampleRecRel_exactly_h(m,h,decimal_precision=10):
    """
    Samples a reciprocal relation in Q_m^{h}(\not CW), where all non-diagonal entries 
    are in {0.5-h , 0.5+h}.
    
    EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = sampleRecRel_exactly_h(5,0.1)
Q.show()
print(has_CW(Q))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """ 
    Q = sampleReciprocal(m,decimal_precision)
    Q = __EnforceBoundedFromOneHalf__(Q,0.4)
    for i in range(0,Q.m):
        for j in range(0,Q.m):
            if Q.Q[i,j]>0.5:
                Q.Q[i,j] = 0.5+h
            if Q.Q[i,j]<0.5:
                Q.Q[i,j] = 0.5-h
    return(Q) 

###############################################################################
#   Function: sampleCW_exactly_h
###############################################################################
def sampleCW_exactly_h(m,h,decimal_precision=10):
    """
    Samples a reciprocal relation in Q_m^{h}(CW), where all non-diagonal entries 
    are in {0.5-h , 0.5+h}.
    
    EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q, buf  = sampleCW_exactly_h(5,0.1)
Q.show()
print("CW of Q:",buf)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """ 
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    Q, buf = sampleCW_boundedFromOneHalf(m,0.4,decimal_precision)
    for i in range(0,Q.m):
        for j in range(0,Q.m):
            if Q.Q[i,j]>0.5:
                Q.Q[i,j] = 0.5+h
            if Q.Q[i,j]<0.5:
                Q.Q[i,j] = 0.5-h
    return(Q,buf) 


###############################################################################
#   Function: sampleNotCW_exactly_h
###############################################################################
def sampleNotCW_exactly_h(m,h,max_tries=1000,decimal_precision=10):
    """
    Samples a reciprocal relation in Q_m^{h}(\not CW), where all non-diagonal entries 
    are in {0.5-h , 0.5+h}.
    
    EXAMPLE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Q = sampleNotCW_exactly_h(5,0.1)
Q.show()
print(has_CW(Q))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """ 
    assert type(h) is float and 0<h<1/2, "The parameter `h` has to be a `float` in the interval :math:`(0,0.5)`"
    Q = sampleNotCW_boundedFromOneHalf(m=m,h=0.4,max_tries=1000,decimal_precision=decimal_precision)
    for i in range(0,Q.m):
        for j in range(0,Q.m):
            if Q.Q[i,j]>0.5:
                Q.Q[i,j] = 0.5+h
            if Q.Q[i,j]<0.5:
                Q.Q[i,j] = 0.5-h
    return(Q)

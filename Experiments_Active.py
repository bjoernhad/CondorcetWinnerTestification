#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the experiments in the active scenario of [Haddenhorst2021].
In order to rerun these, UNCOMMENT in "Part 4" below the corresponding lines and 
then execute this file.

[Haddenhorst2021]: B. Haddenhorst, V. Bengs, J. Brandt and E.Hüllermeier, Testification of Condorcet Winners in Dueling Bandits, Proceedings of UAI, 2021
"""
import ReciprocalRelations as rr
import TestEnvironment as tenv
import TestingComponent as tc
import numpy as np
import DeterministicTestingComponent as dtc
import SELECT as select
import math
import matplotlib.pyplot as plt



nr_items, decimal_precision = 5,3
np.set_printoptions(precision=3)

###############################################################################
#   PART 1: Define 'find_CW_with_symm_test', 'SELECT_then_verify' 
#           and 'buf_SPRT' 
###############################################################################
def find_CW_with_symm_test(TE, h, gamma):
    """
    This is an implementation of the Algorithm NTS from the paper.
    """
    SymmTC = tc.Symmetric_TestingComponent(TE.N, TE.R, h=float(h), gamma=gamma)
    sampling_strategy = dtc.Optimal_Deterministic_CW_Tester(len(TE.N))
    for t in range(500000):
        [i,j] = sampling_strategy.getQuery()
        while not SymmTC.G.has_edge(i,j) and not SymmTC.G.has_edge(j,i):
            feedback = TE.pullArmPair(i,j)
            SymmTC.update(i,j,feedback)
            if SymmTC.TC():
                SymmTC.DC()
                return SymmTC.find_CW(), SymmTC.time
        sampling_strategy.giveFeedback(feedback)

def SELECT_then_verify(TE,h,gamma, variant = "Hoeffding"):
    """
    This is the implementation from SELECT-then-verify from the paper.
    The internal hypothesis test for verifying the output of SELECT can either
    be the non-sequential Hoeffding-bound test (with 'variant="Hoeffding"')
    or the corresponding SPRT (with 'variant="SPRT"').
    """
    assert variant=="Hoeffding" or variant=="SPRT", "'variant' has to be 'Hoeffding' or 'SPRT'"
    m = TE.P.m
    epsilon = -np.log(0.5*gamma)/np.log(np.log2(m))
    m_h = math.floor((1+epsilon)*math.log(2)/2*math.log(math.log(m,2),2)/(h*h))+1
    CW_M, itera = select.select(list(np.arange(m)), m_h, TE.pullArmPair)
    CW_M = int(CW_M)
    # print("Output of SELECT:",CW_M,"(after ",itera," iterations)")
    # TE.show()
    if variant=="Hoeffding":
        t0 = np.ceil(2/(h**2) * np.log(2*(m-1)/gamma))    
        w = np.zeros(m)
        for j in range(0,m):
            if CW_M is not j:
                for t in range(0,int(t0)):
                    buf = TE.pullArmPair(CW_M,j)
                    w[j] += buf     #Increases by 1 if CW_M has won
                    if w[j]/t0 < 0.5:
                        # print("w",w)
                        return(False)
        return(CW_M)
    
    if variant=="SPRT":
        for j in range(0,m):
            if CW_M is not j:
                winner = buf_SPRT(TE,h,gamma/(2*(m-1)),CW_M,j)
                if winner != CW_M:
                    return(False)
    return(CW_M)

def buf_SPRT(TE,h,gamma,i,j):
    """ 
    This function conducts a SPRT (with parameters h,gamma) in order to decide
    whether the (i,j)-entry of TE.P is >1/2 or <1/2.
    """
    N = 1
    C = (1/(2*N)) * np.ceil(np.log( (1-gamma) / gamma ) / np.log( (0.5+h) / (0.5-h) ))
    w = TE.pullArmPair(i,j)
    while 0.5-C < w/N and w/N< 0.5+C:
        w += TE.pullArmPair(i,j)
        C = (1/(2*N)) * np.ceil(np.log( (1-gamma) / gamma ) / np.log( (0.5+h) / (0.5-h) ))
        N = N+1
    if w/N >= 0.5+C: 
        return(i)
    else:
        return(j)
        
###############################################################################
#   PART 3: Define the evaluation functions 'experiments_one', 'experiment_two'
#           'experiment_three' as well as 'generate_main_figure'.
###############################################################################
def experiments_one(m,h,gamma,nr_iterations=100, real_h = 0.05, has_CW = "No",verify_variant="SPRT"):
    """
    This function compares NTS with SELECT-then-verify. It is required for the
    function "experiment_two".
    """
    assert has_CW=="No" or has_CW=="Yes" or has_CW=="Both", "'has_CW' has to be 'Yes','No' or 'Both'."
    results = dict()
    results["NTS_output"] =list()
    results["NTS_time"] = list()
    results["S_t_verify_output"] = list()
    results["S_t_verify_time"] = list()
    results["Truth"] = list()
    for iteration in range(0,nr_iterations):
        # Step 1: Sample a reciprocal relation, create a TE and a dictionary to save results.
        if has_CW == "No":
            P, buf = rr.sampleCW_boundedFromOneHalf(m,real_h,decimal_precision=3)
        elif has_CW == "Yes": 
            P = rr.sampleNotCW_boundedFromOneHalf(m,real_h,max_tries=10000,decimal_precision=3)
        else:
            P = rr.sampleReciprocal(m,decimal_precision=3)
            P = rr.__EnforceBoundedFromOneHalf__(P,real_h)
        results["Truth"].append(rr.get_CW(P))
        
        # Step 2: Run and log our NTS
        TE = tenv.TestEnvironment(P)
        current_output = find_CW_with_symm_test(TE, h, gamma)
        results["NTS_output"].append(current_output[0])
        results["NTS_time"].append(TE.time)
        
        # Step 3: Run and log SELECT_then_verify
        TE = tenv.TestEnvironment(P)
        current_output = SELECT_then_verify(TE, h, gamma,variant=verify_variant)
        results["S_t_verify_output"].append(current_output)
        results["S_t_verify_time"].append(TE.time)
        
    # Step 4: Calculate the accuracy of both algorithms and return the results
    nr_correct_NTS ,nr_correct_S_t_verify = 0,0
    for i in range(0,nr_iterations):
        if results["Truth"][i] == results["NTS_output"][i]:
            nr_correct_NTS += 1
        if results["Truth"][i] == results["S_t_verify_output"][i]:
            nr_correct_S_t_verify += 1
    results["Acc_NTS"] = nr_correct_NTS / nr_iterations
    results["Acc_S_t_verify"] = nr_correct_S_t_verify / nr_iterations
    results["NTS_mean_time"] = np.mean(results["NTS_time"])
    results["S_t_verify_mean_time"] = np.mean(results["S_t_verify_time"])
    # print(results)
    # print("NTS: mean",np.mean(results["NTS_time"]),"\t std:",np.std(results["NTS_time"]), "\t Accuracy:", results["Acc_NTS"])
    # print("SELECT_then_verify: mean",np.mean(results["S_t_verify_time"]),"\t std:",np.std(results["S_t_verify_time"]), "\t Accuracy:", results["Acc_S_t_verify"])    
    return(results)

def experiment_two(m=5, real_h=0.1, h=0.3, file_name ="plot",nr_iterations =25000,has_CW = "No",verify_variant="SPRT"): 
    """ 
    This function compares NTS with SELECT-then-verify for the given parameters,
    saves the observed accuracies and averaged termination times and plots the 
    results.
    """    
    gammas = [0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.35,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.7,0.95,0.99]
    x_NTS = np.zeros(len(gammas))
    y_NTS = np.zeros(len(gammas))
    x_StV = np.zeros(len(gammas))
    y_StV = np.zeros(len(gammas))
    print("Progress for "+str(file_name)+" (.. out of "+str(len(gammas))+"): ")
    for i in range(0,len(gammas)):
        print(i,end=",")
        buf = experiments_one(m,h,gammas[i],nr_iterations,real_h=real_h,has_CW = has_CW)
        x_NTS[i] = buf["NTS_mean_time"]
        x_StV[i] = buf["S_t_verify_mean_time"]
        y_NTS[i] = buf["Acc_NTS"]
        y_StV[i] = buf["Acc_S_t_verify"]    
    plt.plot(x_NTS,y_NTS, marker = "^", label="NTS")
    plt.plot(x_StV, y_StV, marker = "o", label="SELECT-then-verify")
    plt.xlabel("Iterations")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.title("h="+str(h))
    plt.savefig(str(file_name)+"_plot.png",dpi=300)
    plt.show() 
    np.savetxt(str(file_name)+"_results.csv",np.asarray([x_NTS,y_NTS,x_StV,y_StV]),delimiter=",")
    # print(x_NTS,y_NTS)
    print("Done.")

def experiment_three_help(m,h,gamma,nr_iterations=100, real_h = 0.05, has_CW = "No",verify_variant="SPRT"):
    """
    This is an appropriate modification of 'experiments_one' from above
    Each line of code that has been modified is indicated with a comment '[CHANGE]', 
    together with the original code.
    """
    assert has_CW=="No" or has_CW=="Yes" or has_CW=="Both", "'has_CW' has to be 'Yes','No' or 'Both'."
    results = dict()
    results["NTS_output"] =list()
    results["NTS_time"] = list()
    results["S_t_verify_output"] = list()
    results["S_t_verify_time"] = list()
    results["Truth"] = list()
    for iteration in range(0,nr_iterations):
        # Step 1: Sample a reciprocal relation, create a TE and a dictionary to save results.
        if has_CW == "No":
            #P, buf = rr.sampleCW_boundedFromOneHalf(m,real_h,decimal_precision=3)  [CHANGE]
            P, buf = rr.sampleCW_exactly_h(m,real_h,decimal_precision=3)
        elif has_CW == "Yes": 
            # P = rr.sampleNotCW_boundedFromOneHalf(m,real_h,max_tries=10000,decimal_precision=3) [CHANGE]
            P = rr.sampleNotCW_exactly_h(m, real_h,max_tries = 100000,decimal_precision = 3)
        else:
            # P = rr.sampleReciprocal(m,decimal_precision=3) [CHANGE]
            # P = rr.__EnforceBoundedFromOneHalf__(P,real_h) [CHANGE]
            P = rr.sampleRecRel_exactly_h(m,real_h,decimal_precision = 3)
        results["Truth"].append(rr.get_CW(P))
        
        # Step 2: Run and log our NTS
        TE = tenv.TestEnvironment(P)
        current_output = find_CW_with_symm_test(TE, h, gamma)
        results["NTS_output"].append(current_output[0])
        results["NTS_time"].append(TE.time)
        
        # Step 3: Run and log SELECT_then_verify
        TE = tenv.TestEnvironment(P)
        current_output = SELECT_then_verify(TE, h, gamma,variant=verify_variant)
        results["S_t_verify_output"].append(current_output)
        results["S_t_verify_time"].append(TE.time)
        
    # Step 4: Calculate the accuracy of both algorithms and return the results
    nr_correct_NTS ,nr_correct_S_t_verify = 0,0
    for i in range(0,nr_iterations):
        if results["Truth"][i] == results["NTS_output"][i]:
            nr_correct_NTS += 1
        if results["Truth"][i] == results["S_t_verify_output"][i]:
            nr_correct_S_t_verify += 1
    results["Acc_NTS"] = nr_correct_NTS / nr_iterations
    results["Acc_S_t_verify"] = nr_correct_S_t_verify / nr_iterations
    results["NTS_mean_time"] = np.mean(results["NTS_time"])
    results["S_t_verify_mean_time"] = np.mean(results["S_t_verify_time"])
    return(results)


def experiment_three(m=5, gamma = 0.05, real_hs=[0.1,0.2,0.3,0.4], h=0.1, file_name ="plot",nr_iterations =25000,has_CW = "No",verify_variant="SPRT"): 
    """ 
    This is a simple modification of experiment_two. 
    -- instead of modifying gamma, we modify the value of real_h.
    """    
    # gammas = [0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.35,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.7,0.95,0.99]
    len_real_hs = len(real_hs)
    x_NTS = np.zeros(len_real_hs)
    y_NTS = np.zeros(len_real_hs)
    x_StV = np.zeros(len_real_hs)
    y_StV = np.zeros(len_real_hs)
    print("Progress for "+str(file_name)+" (.. out of "+str(len_real_hs)+"): ")
    for i in range(0,len_real_hs):
        print(i,end=",")
        buf = experiment_three_help(m,h,gamma,nr_iterations,real_h=real_hs[i],has_CW = has_CW)
        x_NTS[i] = buf["NTS_mean_time"]
        x_StV[i] = buf["S_t_verify_mean_time"]
        y_NTS[i] = buf["Acc_NTS"]
        y_StV[i] = buf["Acc_S_t_verify"]    
    plt.plot(real_hs,x_NTS, marker = "^", label="NTS")
    plt.plot(real_hs, x_StV, marker = "o", label="SELECT-then-verify")
    plt.xlabel("real_h")
    plt.ylabel("Iterations")
    plt.legend()
    plt.title("h="+str(h))
    plt.savefig(str(file_name)+"_plot.png",dpi=300)
    plt.show() 
    np.savetxt(str(file_name)+"_results.csv",np.asarray([real_hs,x_NTS,y_NTS,x_StV,y_StV]),delimiter=",")
    print("Done.")
    # OUTPUT THE RESULTS AS A TABLE:
    print("The results in form of [h, T A^NTS, Acc. A^NTS, T^StV, Acc StV] are:")
    for i in range(0,len_real_hs):
        print(real_hs[i],x_NTS[i],y_NTS[i],x_StV[i],y_StV[i])
        

##############################################################################
# The following function helps to create Figure 1  of our paper.
##############################################################################
def generate_main_figure():
    a = np.loadtxt("MAIN_h02_results.csv",delimiter=",")
    b = np.loadtxt("MAIN_h03_results.csv",delimiter=",")
    plt.figure(figsize=(12,5))
    plt.rcParams.update({'font.size': 14})
    plt.subplot(1,2,1)
    plt.subplots_adjust(left=0.07, 
                    bottom=0.1,  
                    right=0.99,  
                    top=0.92,  
                    wspace=0.1,  
                    hspace=0.1) 
    plt.plot(a[0], a[1], marker = "^", label="NTS")
    plt.plot(a[2], a[3], marker = "o", label="SELECT-then-verify")
    plt.ylabel("Success Rate")
    plt.xlabel("Iterations")
    plt.title("h=0.2")
    plt.legend(loc="lower right")
    plt.subplot(1,2,2)
    plt.xlim(left=0.5*min(min(b[0]),min(b[2])),right=2*min(max(b[0]),max(b[2])))
    plt.plot(b[0],b[1], marker = "^", label="NTS")
    plt.plot(b[2],b[3], marker = "o", label="SELECT-then-verify")
    plt.xlabel("Iterations")
    plt.title("h=0.3")
    plt.legend(loc="lower right")
    plt.savefig("MAIN_figure.png",dpi=600)
    # fig.tight_layout()     
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    




###############################################################################
#   PART 4: Reconstruct the results from our paper.
###############################################################################
    
###############################################################################
#   PART 4.1: Reconstruct the results from Section 7.1
###############################################################################
NR_it = 25000
np.random.seed(1)
experiment_two(m=5, real_h=0.05, h=0.2, file_name ="MAIN_h02",nr_iterations =NR_it,has_CW = "Both",verify_variant="SPRT")
np.random.seed(2)
experiment_two(m=5, real_h=0.05, h=0.3, file_name ="MAIN_h03",nr_iterations =NR_it,has_CW = "Both",verify_variant="SPRT")
generate_main_figure()


###############################################################################
#   PART 4.2.1: Reconstruct the results from Section I.1, Figure 3
#               (Similar to those in Sec. 7.1, but with larger number of arms)
###############################################################################
NR_it = 100000
np.random.seed(3)
experiment_two(m=10, real_h=0.05, h=0.3, file_name ="SUPPL_LARGE10_h03",nr_iterations =NR_it,has_CW = "Both",verify_variant="SPRT")
np.random.seed(101)
experiment_two(m=8, real_h=0.05, h=0.3, file_name ="SUPPL_LARGE08_h03",nr_iterations =NR_it,has_CW = "Both",verify_variant="SPRT")

###############################################################################
#   PART 4.2.2: Reconstruct the results from Section I.1, Table 2
#               (Comparison of SELECT-then-Verify and NTS on \hat{Q}_{m}^{h})
###############################################################################
NR_it = 100
real_hs = [0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]    
np.random.seed(3)
experiment_three(m=20, gamma = 0.05, real_hs=real_hs, h=0.05, file_name ="Exp3_m20_h05",nr_iterations =NR_it,has_CW = "Yes",verify_variant="SPRT")

###############################################################################
#   PART 4.2.2: Reconstruct the results from Section I.2, Figures 4 & 5
#               (Similar to those in Sec. 7.1, but with restriction to CW/Non-CW instances)
###############################################################################
NR_it = 25000
np.random.seed(5)
experiment_two(m=5, real_h=0.05, h=0.1, file_name ="SUPPL_CW_h01",nr_iterations =NR_it,has_CW = "Yes",verify_variant="SPRT")
np.random.seed(6)
experiment_two(m=5, real_h=0.05, h=0.2, file_name ="SUPPL_CW_h02",nr_iterations =NR_it,has_CW = "Yes",verify_variant="SPRT")
np.random.seed(7)
experiment_two(m=5, real_h=0.05, h=0.3, file_name ="SUPPL_CW_h03",nr_iterations =NR_it,has_CW = "Yes",verify_variant="SPRT")
np.random.seed(8)
experiment_two(m=5, real_h=0.05, h=0.1, file_name ="SUPPL_noCW_h01",nr_iterations =NR_it,has_CW = "No",verify_variant="SPRT")
np.random.seed(9)
experiment_two(m=5, real_h=0.05, h=0.2, file_name ="SUPPL_noCW_h02",nr_iterations =NR_it,has_CW = "No",verify_variant="SPRT")
np.random.seed(10)
experiment_two(m=5, real_h=0.05, h=0.3, file_name ="SUPPL_noCW_h03",nr_iterations =NR_it,has_CW = "No",verify_variant="SPRT")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains a class "Regret", which updates and stores the
following different regret types:
    - cumulative regret
    - weak regret
    - strong regret
    - Copeland regret
    - Borda regret
"""
from ReciprocalRelations import ReciprocalRelation
import numpy as np


###############################################################################
# Class: Regret
###############################################################################
class Regret(object):
    """
    Class for computing, updating and storing different regret types like
    cumulative regret, weak regret, strong regret, copeland regret, borda regret
    """

    def __init__(self, Q):
        assert type(Q) is ReciprocalRelation
        self.Q = Q.Q
        self.borda_regret = []
        self.copeland_regret = []
        self.weak_regret = []
        self.strong_regret = []
        self.cumulative_regret = []
        self.condorcet_winner = self.get_condorcet_winner()
        self.borda_winner = self.get_borda_winner()
        self.copeland_winner = self.get_copeland_winner()

    def get_condorcet_winner(self):
        """
        Get Condorcet winner of win probability matrix Q if it exists
        """
        for arm1 in range(len(self.Q)):
            CW = True
            for arm2 in range(len(self.Q)):
                if arm1 != arm2:
                    if self.Q[arm1, arm2] < 0.5:
                        # line is definitiv not the CW!
                        CW = False
                        continue
            if CW == True:
                # we have found a CW!
                return arm1
        # no CW exists
        return None

    def get_borda_winner(self):
        """
        Computes Borda winner for reciprocal relation Q
        """
        borda_score = [np.sum(self.Q[i]) for i in range(len(self.Q))]
        borda_score = np.array(borda_score)
        return list(np.where(borda_score == np.max(borda_score))[0])[0]

    def get_copeland_winner(self):
        """
        Computes Copeland winner for reciprocal relation Q
        """
        copeland_score = [len(np.where(self.Q[i] > 0.5)[0]) for i in range(len(self.Q))]
        copeland_score = np.array(copeland_score)
        return list(np.where(copeland_score == np.max(copeland_score))[0])[0]

    def update_weak_regret(self, i, j):
        """
        updates weak regret for pulled arms i and j if condorcet winner exists
        """
        assert type(i) is int and 0 <= i and i <= len(self.Q)
        assert type(j) is int and 0 <= j and j <= len(self.Q)
        assert self.condorcet_winner is not None
        if self.weak_regret == []:
            self.weak_regret = [min(self.Q[self.condorcet_winner][i], self.Q[self.condorcet_winner][j])]
        else:
            self.weak_regret.append(self.weak_regret[-1] + min(self.Q[self.condorcet_winner][i], self.Q[self.condorcet_winner][j]))

    def update_strong_regret(self, i, j):
        """
        updates strong regret for pulled arms i and j if condorcet winner exists
        """
        assert type(i) is int and 0 <= i and i <= len(self.Q)
        assert type(j) is int and 0 <= j and j <= len(self.Q)
        assert self.condorcet_winner is not None
        if self.strong_regret == []:
            self.strong_regret = [max(self.Q[self.condorcet_winner][i], self.Q[self.condorcet_winner][j])]
        else:
            self.strong_regret.append(self.strong_regret[-1] + max(self.Q[self.condorcet_winner][i], self.Q[self.condorcet_winner][j]))

    def update_cumulative_regret(self, i, j):
        """
        updates cumulative regret for pulled arms i and j if condorcet winner exists
        """
        assert type(i) is int and 0 <= i and i <= len(self.Q)
        assert type(j) is int and 0 <= j and j <= len(self.Q)
        assert self.condorcet_winner is not None
        if self.cumulative_regret == []:
            self.cumulative_regret = [0.5*(self.Q[self.condorcet_winner][i] + self.Q[self.condorcet_winner][j])]
        else:
            self.cumulative_regret.append(self.cumulative_regret[-1] + 0.5*(self.Q[self.condorcet_winner][i] + self.Q[self.condorcet_winner][j]))

    def copeland_score(self, i):
        """
        Computes copeland score for arm i
        """
        return len(np.where(self.Q[i] > 0.5)[0])

    def update_copeland_regret(self, i, j):
        """
        updates copeland regret for pulled arms i and j if condorcet winner exists
        """
        assert type(i) is int and 0 <= i and i <= len(self.Q)
        assert type(j) is int and 0 <= j and j <= len(self.Q)
        assert self.copeland_winner is not None
        if self.copeland_regret == []:
            self.copeland_regret = [0.5*(self.copeland_score(self.copeland_winner) - self.copeland_score(i)
                                         + self.copeland_score(self.copeland_winner) - self.copeland_score(j))]
        else:
            self.copeland_regret.append(self.copeland_regret[-1] + 0.5 * (self.copeland_score(self.copeland_winner) - self.copeland_score(i)
                                           + self.copeland_score(self.copeland_winner) - self.copeland_score(j)))

    def borda_score(self, i):
        """
        Computes copeland score for arm i
        """
        return np.sum(self.Q[i])

    def update_borda_regret(self, i, j):
        """
        updates borda regret for pulled arms i and j if condorcet winner exists
        """
        assert type(i) is int and 0 <= i and i <= len(self.Q)
        assert type(j) is int and 0 <= j and j <= len(self.Q)
        assert self.borda_winner is not None
        if self.borda_regret == []:
            self.borda_regret = [0.5*(self.borda_score(self.borda_winner) - self.borda_score(i)
                                         + self.borda_score(self.borda_winner) - self.borda_score(j))]
        else:
            self.borda_regret.append(self.borda_regret[-1] + 0.5 * (self.borda_score(self.borda_winner) - self.borda_score(i)
                                           + self.borda_score(self.borda_winner) - self.borda_score(j)))


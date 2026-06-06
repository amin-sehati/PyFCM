# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt
import xlrd
import numpy as np
import math
import networkx as nx


def _transform_func(x, n, f_type, landa=1):
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-landa * x[i]))
        return x_new
    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = math.tanh(landa * x[i])
        return x_new
    if f_type == "bivalent":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 if x[i] > 0 else 0
        return x_new
    if f_type == "trivalent":
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            elif x[i] == 0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        return x_new


def _infer_steady(Adj_matrix, n, init_vec, f_type, infer_rule, landa):
    act_vec_old = init_vec.copy()
    AdjmT = Adj_matrix.T
    resid = 1
    while resid > 0.00001:
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(AdjmT, (2 * act_vec_old - np.ones(n)))
        act_vec_new = _transform_func(x, n, f_type, landa)
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def _infer_scenario(Scenario_concept, Adj_matrix, n, init_vec, f_type, infer_rule, landa, changeLevel=1):
    act_vec_old = init_vec.copy()
    AdjmT = Adj_matrix.T
    resid = 1
    while resid > 0.00001:
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(AdjmT, (2 * act_vec_old - np.ones(n)))
        act_vec_new = _transform_func(x, n, f_type, landa)
        act_vec_new[Scenario_concept] = changeLevel
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def run_sensitivity(file_location, noise_threshold, infer_rule, function_type, lambda_,
                    principles, scenario_variables):
    """
    Run FCM sensitivity analysis: vary each scenario variable from 0 to 1 and
    plot how the system principles respond.

    Parameters
    ----------
    file_location : str
        Path to the single-participant adjacency matrix Excel file.
    noise_threshold : float
        Edges with |weight| <= noise_threshold are zeroed out.
    infer_rule : str
        'k', 'mk', or 'r'.
    function_type : str
        'sig', 'tanh', 'bivalent', or 'trivalent'.
    lambda_ : float
        Squashing function steepness parameter.
    principles : list of str
        Names of the key output concepts to track.
    scenario_variables : list of str
        Concept names to sweep as input variables.
    """
    workbook = xlrd.open_workbook(file_location)
    sheet = workbook.sheet_by_index(0)
    n_concepts = sheet.nrows - 1

    Adj_matrix = np.zeros((n_concepts, n_concepts))
    activation_vec = np.ones(n_concepts)
    node_name = {}

    for i in range(1, n_concepts + 1):
        for j in range(1, n_concepts + 1):
            val = sheet.cell_value(i, j)
            Adj_matrix[i - 1, j - 1] = 0 if abs(val) <= noise_threshold else val

    Concepts_matrix = [sheet.cell_value(0, i) for i in range(1, n_concepts + 1)]
    G = nx.DiGraph(Adj_matrix)
    for nod in G.nodes():
        node_name[nod] = sheet.cell_value(nod + 1, 0)

    prin_concepts_index = [nod for nod in node_name if node_name[nod] in principles]

    SteadyState = _infer_steady(Adj_matrix, n_concepts, activation_vec, function_type, infer_rule, lambda_)

    for name in scenario_variables:
        Scenario_concept = Concepts_matrix.index(name)
        change_levels = np.linspace(0, 1, 21)

        change_in_principles = {pr: [] for pr in prin_concepts_index}

        for c in change_levels:
            ScenarioState = _infer_scenario(Scenario_concept, Adj_matrix, n_concepts, activation_vec,
                                            function_type, infer_rule, lambda_, changeLevel=c)
            changes = ScenarioState - SteadyState
            for pr in prin_concepts_index:
                change_in_principles[pr].append(changes[pr])

        plt.clf()
        for pr in prin_concepts_index:
            plt.plot(change_levels, change_in_principles[pr], '-o', markersize=3, label=node_name[pr])
        plt.legend(fontsize=8)
        plt.xlabel("activation state of {}".format(name))
        plt.ylabel('State of system principles')
        plt.savefig('{}.pdf'.format(name))
        plt.show()

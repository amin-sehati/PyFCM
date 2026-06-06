# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xlrd


def _transform_func(x, n, f_type, lambda_=1):
    if f_type == "sig":
        x_new = np.empty(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-lambda_ * x[i]))
        return x_new
    if f_type == "tanh":
        x_new = np.empty(n)
        for i in range(n):
            x_new[i] = math.tanh(lambda_ * x[i])
        return x_new
    if f_type == "bivalent":
        x_new = np.empty(n)
        for i in range(n):
            x_new[i] = 1 if x[i] > 0 else 0
        return x_new
    if f_type == "trivalent":
        x_new = np.empty(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            elif x[i] == 0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        return x_new


def _infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    act_vec_old = init_vec.copy()
    adj_matrix_t = adj_matrix.T
    resid = 1
    ones = np.ones(n) if infer_rule == "r" else None
    while resid > 0.00001:
        if infer_rule == "k":
            x = np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "mk":
            x = act_vec_old + np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "r":
            shifted_state = 2 * act_vec_old - ones
            x = shifted_state + np.matmul(adj_matrix_t, shifted_state)
        else:
            x = np.zeros(n)
        act_vec_new = _transform_func(x, n, f_type, lambda_)
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def _infer_scenario(scenario_concept, adj_matrix, n, init_vec, f_type, infer_rule, lambda_, change_level=1):
    act_vec_old = init_vec.copy()
    adj_matrix_t = adj_matrix.T
    resid = 1
    ones = np.ones(n) if infer_rule == "r" else None
    while resid > 0.00001:
        if infer_rule == "k":
            x = np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "mk":
            x = act_vec_old + np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "r":
            shifted_state = 2 * act_vec_old - ones
            x = shifted_state + np.matmul(adj_matrix_t, shifted_state)
        else:
            x = np.zeros(n)
        act_vec_new = _transform_func(x, n, f_type, lambda_)
        act_vec_new[scenario_concept] = change_level
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

    adj_matrix = np.zeros((n_concepts, n_concepts))
    activation_vec = np.ones(n_concepts)
    node_name = {}

    for i in range(1, n_concepts + 1):
        for j in range(1, n_concepts + 1):
            weight = sheet.cell_value(i, j)
            adj_matrix[i - 1, j - 1] = 0 if abs(weight) <= noise_threshold else weight

    concepts = [sheet.cell_value(0, i) for i in range(1, n_concepts + 1)]
    G = nx.DiGraph(adj_matrix)
    for nod in G.nodes():
        node_name[nod] = sheet.cell_value(nod + 1, 0)

    prin_concepts_index = [nod for nod in node_name if node_name[nod] in principles]

    steady_state = _infer_steady(adj_matrix, n_concepts, activation_vec, function_type, infer_rule, lambda_)

    for name in scenario_variables:
        scenario_concept = concepts.index(name)
        change_levels = np.linspace(0, 1, 21)

        change_in_principles = {pr: [] for pr in prin_concepts_index}

        for c in change_levels:
            scenario_state = _infer_scenario(scenario_concept, adj_matrix, n_concepts, activation_vec,
                                             function_type, infer_rule, lambda_, change_level=c)
            changes = scenario_state - steady_state
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

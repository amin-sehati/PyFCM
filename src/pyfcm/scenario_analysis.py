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


def _transform_func(x, n, f_type, Lambda=1):
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-Lambda * x[i]))
        return x_new
    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = math.tanh(Lambda * x[i])
        return x_new
    if f_type == "biv":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 if x[i] > 0 else 0
        return x_new
    if f_type == "triv":
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            elif x[i] == 0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        return x_new


def _infer_steady(Adj_matrix, n, init_vec, f_type, infer_rule, Lambda):
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
        act_vec_new = _transform_func(x, n, f_type, Lambda)
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def _infer_scenario(Scenario_concepts, change_level, Adj_matrix, n, init_vec, f_type, infer_rule, Lambda):
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
        act_vec_new = _transform_func(x, n, f_type, Lambda)
        for c in Scenario_concepts:
            act_vec_new[c] = change_level[c]
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def run_scenario(file_location, noise_threshold, infer_rule, function_type, lambda_,
                 principles, change_levels, show='A'):
    """
    Run an FCM scenario analysis.

    Parameters
    ----------
    file_location : str
        Path to the single-participant adjacency matrix Excel file.
    noise_threshold : float
        Edges with |weight| <= noise_threshold are zeroed out.
    infer_rule : str
        'k', 'mk', or 'r'.
    function_type : str
        'sig', 'tanh', 'biv', or 'triv'.
    lambda_ : float
        Squashing function steepness parameter.
    principles : list of str
        Names of the key output concepts to track.
    change_levels : dict
        {concept_name: activation_level} for each scenario variable.
    show : str
        'A' to plot all concepts, 'P' to plot principles only.

    Returns
    -------
    changes_dic : dict  {concept_name: change_value}
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

    change_level_by_index = {Concepts_matrix.index(name): lvl for name, lvl in change_levels.items()}
    Scenario_concepts = [Concepts_matrix.index(name) for name in change_levels]

    SteadyState = _infer_steady(Adj_matrix, n_concepts, activation_vec, function_type, infer_rule, lambda_)
    ScenarioState = _infer_scenario(Scenario_concepts, change_level_by_index, Adj_matrix,
                                    n_concepts, activation_vec, function_type, infer_rule, lambda_)
    change_IN_ALL = ScenarioState - SteadyState

    for c in Scenario_concepts:
        change_IN_ALL[c] = 0

    change_IN_PRINCIPLES = [change_IN_ALL[i] for i in prin_concepts_index]

    if show == "A":
        changes = change_IN_ALL
        plt.figure(figsize=(50, 5))
        plt.bar(np.arange(len(changes)), changes, align='center', alpha=1, color='g')
        plt.xticks(np.arange(len(changes)), Concepts_matrix, rotation='vertical')
    else:
        changes = change_IN_PRINCIPLES
        plt.figure(figsize=(10, 3))
        plt.bar(np.arange(len(changes)), changes, align='center', alpha=1, color='b')
        plt.xticks(np.arange(len(changes)), principles, rotation='vertical')

    plt.title("changes in variables")
    ax = plt.axes()
    ax.xaxis.grid()
    plt.savefig('Scenario_Results.pdf')
    plt.show()

    changes_dic = {node_name[nod]: change_IN_ALL[nod] for nod in G.nodes()}
    with open('Changes_In_All_Concepts.csv', 'w') as f:
        for key, value in changes_dic.items():
            f.write('{},{}\n'.format(key, value))

    return changes_dic

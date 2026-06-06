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
import random
import pandas as pd
import networkx as nx
from math import pi


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
        if resid < 0.00001:
            break
        act_vec_old = act_vec_new
    return act_vec_new


def _infer_scenario(Scenario_concepts, zeros, Adj_matrix, n, init_vec, f_type, infer_rule, landa):
    act_vec_old = init_vec.copy()
    AdjmT = Adj_matrix.T

    my_random = {rC: random.random() * random.choice([-1, 1]) for rC in Scenario_concepts}

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
        for z in zeros:
            act_vec_new[z] = 0
        for c in Scenario_concepts:
            act_vec_new[c] = my_random[c]
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def run_uncertainty(file_location, noise_threshold, infer_rule, function_type, lambda_,
                    principles, thresh, n_iteration):
    """
    Run FCM uncertainty analysis via Monte Carlo sampling of input combinations.

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
    thresh : int
        Maximum in-degree for a concept to be eligible as a randomly activated node.
    n_iteration : int
        Number of Monte Carlo iterations.

    Returns
    -------
    df : pd.DataFrame  with columns [IDS, <principle_names...>]
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

    listPossibleNodes = [
        nod for nod in G.nodes()
        if G.in_degree(nod) <= thresh and Concepts_matrix[nod] not in principles
    ]

    SteadyState = _infer_steady(Adj_matrix, n_concepts, activation_vec, function_type, infer_rule, lambda_)

    change_in_principles = {pr: [] for pr in prin_concepts_index}
    iteration = 0

    for _ in range(n_iteration):
        rand = random.randint(1, len(listPossibleNodes))
        Scenario_concepts = random.sample(listPossibleNodes, rand)
        iteration += 1

        ScenarioState = _infer_scenario(Scenario_concepts, listPossibleNodes, Adj_matrix,
                                        n_concepts, activation_vec, function_type, infer_rule, lambda_)
        changes = ScenarioState - SteadyState
        for pr in prin_concepts_index:
            change_in_principles[pr].append(changes[pr])

    df = pd.DataFrame()
    df["IDS"] = list(range(iteration))
    for pr in prin_concepts_index:
        df[node_name[pr]] = change_in_principles[pr]

    categories = list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='black', size=9)
    ax.set_rlabel_position(0)
    plt.yticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"], color="red", size=10)

    for i in range(int(iteration / 10)):
        values = df.loc[i * 10].drop('IDS').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=0.1, color="black", alpha=0.1, linestyle='-')

    plt.savefig('Uncertainty_Analysis_Results.pdf')
    plt.show()

    return df

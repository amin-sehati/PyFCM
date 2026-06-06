# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import math
import random
from math import pi

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xlrd


def _transform_func(x, n, f_type, lambda_=1):
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-lambda_ * x[i]))
        return x_new
    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = math.tanh(lambda_ * x[i])
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


def _infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    act_vec_old = init_vec.copy()
    adj_matrix_t = adj_matrix.T
    resid = 1
    while resid > 0.00001:
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(adj_matrix_t, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(adj_matrix_t, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(adj_matrix_t, (2 * act_vec_old - np.ones(n)))
        act_vec_new = _transform_func(x, n, f_type, lambda_)
        resid = max(abs(act_vec_new - act_vec_old))
        if resid < 0.00001:
            break
        act_vec_old = act_vec_new
    return act_vec_new


def _infer_scenario(scenario_concepts, zero_concepts, adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    act_vec_old = init_vec.copy()
    adj_matrix_t = adj_matrix.T

    random_levels = {concept: random.random() * random.choice([-1, 1]) for concept in scenario_concepts}

    resid = 1
    while resid > 0.00001:
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(adj_matrix_t, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(adj_matrix_t, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(adj_matrix_t, (2 * act_vec_old - np.ones(n)))
        act_vec_new = _transform_func(x, n, f_type, lambda_)
        for z in zero_concepts:
            act_vec_new[z] = 0
        for c in scenario_concepts:
            act_vec_new[c] = random_levels[c]
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

    possible_nodes = [
        nod for nod in G.nodes()
        if G.in_degree(nod) <= thresh and concepts[nod] not in principles
    ]

    steady_state = _infer_steady(adj_matrix, n_concepts, activation_vec, function_type, infer_rule, lambda_)

    change_in_principles = {pr: [] for pr in prin_concepts_index}
    iteration = 0

    for _ in range(n_iteration):
        sample_size = random.randint(1, len(possible_nodes))
        scenario_concepts = random.sample(possible_nodes, sample_size)
        iteration += 1

        scenario_state = _infer_scenario(scenario_concepts, possible_nodes, adj_matrix,
                                        n_concepts, activation_vec, function_type, infer_rule, lambda_)
        changes = scenario_state - steady_state
        for pr in prin_concepts_index:
            change_in_principles[pr].append(changes[pr])

    df = pd.DataFrame()
    df["IDS"] = list(range(iteration))
    for pr in prin_concepts_index:
        df[node_name[pr]] = change_in_principles[pr]

    categories = list(df)[1:]
    n_categories = len(categories)
    angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
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

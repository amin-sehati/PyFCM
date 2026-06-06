# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import random
from math import pi

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .simulation import infer_scenario, infer_steady, transform_func
from .workbooks import concept_metadata, read_single_fcm


def _transform_func(x, n, f_type, lambda_=1):
    return transform_func(x, n, f_type, lambda_)


def _infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    return infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_)


def _infer_scenario(scenario_concepts, zero_concepts, adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    random_levels = {concept: random.random() * random.choice([-1, 1]) for concept in scenario_concepts}
    return infer_scenario(scenario_concepts, random_levels, adj_matrix, n, init_vec,
                          f_type, infer_rule, lambda_, zero_concepts=zero_concepts)


def run_uncertainty(file_location, noise_threshold, infer_rule, function_type, lambda_,
                    principles, thresh, n_iteration):
    """
    Run FCM uncertainty analysis via Monte Carlo sampling of input combinations.

    Parameters
    ----------
    file_location : str
        Path to the single-participant adjacency matrix .xls workbook.
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
    adj_matrix, sheet, n_concepts = read_single_fcm(file_location, noise_threshold)
    activation_vec = np.ones(n_concepts)
    concepts, node_name, prin_concepts_index = concept_metadata(sheet, n_concepts, principles)
    G = nx.DiGraph(adj_matrix)

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

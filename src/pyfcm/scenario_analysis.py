# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt
import numpy as np

from .simulation import infer_scenario, infer_steady, transform_func
from .workbooks import concept_metadata, read_single_fcm


def _transform_func(x, n, f_type, lambda_=1):
    return transform_func(x, n, f_type, lambda_)


def _infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    return infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_)


def _infer_scenario(scenario_concepts, change_level, adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    return infer_scenario(scenario_concepts, change_level, adj_matrix, n, init_vec,
                          f_type, infer_rule, lambda_)


def run_scenario(file_location, noise_threshold, infer_rule, function_type, lambda_,
                 principles, change_levels, show='A'):
    """
    Run an FCM scenario analysis.

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
    change_levels : dict
        {concept_name: activation_level} for each scenario variable.
    show : str
        'A' to plot all concepts, 'P' to plot principles only.

    Returns
    -------
    changes_dic : dict  {concept_name: change_value}
    """
    adj_matrix, sheet, n_concepts = read_single_fcm(file_location, noise_threshold)
    activation_vec = np.ones(n_concepts)
    concepts, node_name, prin_concepts_index = concept_metadata(sheet, n_concepts, principles)

    change_level_by_index = {concepts.index(name): lvl for name, lvl in change_levels.items()}
    scenario_concepts = [concepts.index(name) for name in change_levels]

    steady_state = _infer_steady(adj_matrix, n_concepts, activation_vec, function_type, infer_rule, lambda_)
    scenario_state = _infer_scenario(scenario_concepts, change_level_by_index, adj_matrix,
                                    n_concepts, activation_vec, function_type, infer_rule, lambda_)
    all_changes = scenario_state - steady_state

    for c in scenario_concepts:
        all_changes[c] = 0

    principle_changes = [all_changes[i] for i in prin_concepts_index]

    if show == "A":
        changes = all_changes
        plt.figure(figsize=(50, 5))
        plt.bar(np.arange(len(changes)), changes, align='center', alpha=1, color='g')
        plt.xticks(np.arange(len(changes)), concepts, rotation='vertical')
    else:
        changes = principle_changes
        plt.figure(figsize=(10, 3))
        plt.bar(np.arange(len(changes)), changes, align='center', alpha=1, color='b')
        plt.xticks(np.arange(len(changes)), principles, rotation='vertical')

    plt.title("changes in variables")
    ax = plt.axes()
    ax.xaxis.grid()
    plt.savefig('Scenario_Results.pdf')
    plt.show()

    changes_dic = {node_name[nod]: all_changes[nod] for nod in node_name}
    with open('Changes_In_All_Concepts.csv', 'w') as f:
        for key, value in changes_dic.items():
            f.write('{},{}\n'.format(key, value))

    return changes_dic

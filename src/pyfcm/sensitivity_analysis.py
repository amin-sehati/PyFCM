# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt
import numpy as np

from .simulation import infer_scenario, infer_steady, make_initial_state, transform_func
from .workbooks import concept_metadata, read_single_fcm


def _transform_func(x, n, f_type, lambda_=1):
    return transform_func(x, n, f_type, lambda_)


def _infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_):
    return infer_steady(adj_matrix, n, init_vec, f_type, infer_rule, lambda_)


def _infer_scenario(scenario_concept, adj_matrix, n, init_vec, f_type, infer_rule, lambda_, change_level=1):
    return infer_scenario([scenario_concept], {scenario_concept: change_level},
                          adj_matrix, n, init_vec, f_type, infer_rule, lambda_)


def run_sensitivity(file_location, noise_threshold, infer_rule, function_type, lambda_,
                    principles, scenario_variables, initial_state=1):
    """
    Run FCM sensitivity analysis: vary each scenario variable from 0 to 1 and
    plot how the system principles respond.

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
    scenario_variables : list of str
        Concept names to sweep as input variables.
    initial_state : float or sequence
        Initial activation state. A scalar is applied to all concepts.
    """
    adj_matrix, sheet, n_concepts = read_single_fcm(file_location, noise_threshold)
    activation_vec = make_initial_state(initial_state, n_concepts)
    concepts, node_name, prin_concepts_index = concept_metadata(sheet, n_concepts, principles)

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

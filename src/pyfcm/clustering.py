# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from .aggregation import aggregate_matrices
from .simulation import infer_scenario, infer_steady, transform_func
from .workbooks import read_participant_fcms


def _transform_func(x, n, f_type, lambda_=1):
    return transform_func(x, n, f_type, lambda_)


def _infer_steady(adj_matrix, n, init_vec=None, f_type="tanh", infer_rule="k"):
    return infer_steady(adj_matrix, n, init_vec, f_type, infer_rule)


def _infer_scenario(scenario_concepts, level, adj_matrix, n, init_vec=None, f_type="tanh", infer_rule="k"):
    return infer_scenario(scenario_concepts, level, adj_matrix, n, init_vec,
                          f_type, infer_rule, tolerance=0.0001)


def _similarity(agent_fcm, reference_fcm):
    """Spectral similarity between two FCMs."""
    def select_k(spectrum, minimum_energy=0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)

    laplacian1 = nx.laplacian_spectrum(agent_fcm.to_undirected())
    laplacian2 = nx.laplacian_spectrum(reference_fcm.to_undirected())
    k = min(select_k(laplacian1), select_k(laplacian2))
    return sum((laplacian1[:k] - laplacian2[:k]) ** 2)


def cluster(file_location, aggregation_technique, clustering_method, n_clusters,
            f_type="tanh", infer_rule="k", function_type=None):
    """
    Cluster individual FCMs by structural or dynamic similarity.

    Parameters
    ----------
    file_location : str
        Path to .xls workbook with one participant adjacency matrix per sheet.
    aggregation_technique : str
        How to build the reference FCM: 'AMI', 'AMX', 'O', or 'Z'.
    clustering_method : str
        'S' for structural (spectral) similarity, 'D' for dynamic similarity.
    n_clusters : int
    function_type : str
        Squashing function for dynamic clustering: 'sig', 'tanh', 'bivalent', 'trivalent'.
    f_type : str
        Backward-compatible alias for function_type.
    infer_rule : str
        Inference rule for dynamic clustering: 'k', 'mk', or 'r'.

    Returns
    -------
    clusters : dict  {cluster_id: [similarity_values]}
    assignments : list of (participant_id, cluster_label) tuples
    """
    if function_type is not None:
        f_type = function_type

    all_participants, participant_ids, matrices, _, n_concepts = read_participant_fcms(file_location)

    class Agent:
        def __init__(self, ID):
            self.ID = ID
            self.FCM = nx.DiGraph(all_participants[ID])

    def dynamic(agent, reference_fcm_arr):
        distance = 0
        distances = []
        agent_nodes = list(agent.FCM.nodes())
        steady_state = _infer_steady(all_participants[agent.ID], n_concepts, f_type=f_type, infer_rule=infer_rule)
        reference_steady_state = _infer_steady(reference_fcm_arr, n_concepts, f_type=f_type, infer_rule=infer_rule)
        iteration = 0
        for _ in range(10):
            for _ in range(100):
                sample_size = random.randint(1, n_concepts)
                scenario_concepts = random.sample(agent_nodes, sample_size)
                scenario_levels = {rC: random.random() * random.choice([-1, 1]) for rC in scenario_concepts}
                iteration += 1
                scenario_state = _infer_scenario(scenario_concepts, scenario_levels, all_participants[agent.ID], n_concepts,
                                                f_type=f_type, infer_rule=infer_rule)
                reference_scenario_state = _infer_scenario(scenario_concepts, scenario_levels, reference_fcm_arr, n_concepts,
                                                    f_type=f_type, infer_rule=infer_rule)
                change = scenario_state - steady_state
                reference_change = reference_scenario_state - reference_steady_state
                distance += sum((change - reference_change) ** 2)
            distance = math.sqrt(distance) / iteration
            distances.append(distance)
        return np.mean(distances)

    def make_reference(method):
        if method in ("AMI", "AI", "AMX", "AX"):
            return aggregate_matrices(matrices, method)
        if method == "O":
            return np.ones((n_concepts, n_concepts))
        if method == "Z":
            return np.zeros((n_concepts, n_concepts))

    agents = [Agent(ID=Id) for Id in participant_ids]
    reference_fcm = make_reference(aggregation_technique)

    similarities = {}
    if clustering_method == "D":
        for agent in agents:
            similarities[agent.ID] = dynamic(agent, reference_fcm)
    if clustering_method == "S":
        for agent in agents:
            similarities[agent.ID] = _similarity(agent.FCM, nx.DiGraph(reference_fcm))

    similarity_values = np.array(list(similarities.values()))
    km = KMeans(n_clusters=n_clusters)
    km.fit(similarity_values.reshape(-1, 1))
    assignments = list(zip(list(similarities.keys()), km.labels_))

    clusters = {i: [] for i in range(n_clusters)}
    for participant_id, label in assignments:
        print(participant_id, "is in cluster {}".format(label))
        clusters[label].append(similarities[participant_id])

    plt.figure(figsize=(10, 3))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=0)
    for cl in range(n_clusters):
        plt.plot(clusters[cl], np.zeros_like(clusters[cl]), 'x', markersize=8, label=cl)
    plt.legend()
    plt.savefig('Clusters.pdf')
    plt.show()

    return clusters, assignments

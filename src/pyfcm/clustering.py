# -*- coding: utf-8 -*-
"""
@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt
import xlrd
import numpy as np
import networkx as nx
import math
import random
from sklearn.cluster import KMeans


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


def _infer_steady(Adj, n, init_vec=None, f_type="tanh", infer_rule="k"):
    if init_vec is None:
        init_vec = np.ones(n)
    act_vec_old = init_vec.copy()
    AdjmT = Adj.T
    resid = 1
    while resid > 0.00001:
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(AdjmT, (2 * act_vec_old - np.ones(n)))
        act_vec_new = _transform_func(x, n, f_type)
        resid = max(abs(act_vec_new - act_vec_old))
        if resid < 0.00001:
            break
        act_vec_old = act_vec_new
    return act_vec_new


def _infer_scenario(Scenario_concepts, level, Adj, n, init_vec=None, f_type="tanh", infer_rule="k"):
    if init_vec is None:
        init_vec = np.ones(n)
    act_vec_old = init_vec.copy()
    AdjmT = Adj.T
    resid = 1
    while resid > 0.0001:
        x = np.zeros(n)
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2 * act_vec_old - np.ones(n)) + np.matmul(AdjmT, (2 * act_vec_old - np.ones(n)))
        act_vec_new = _transform_func(x, n, f_type)
        for c in Scenario_concepts:
            act_vec_new[c] = level[c]
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new


def _similarity(agent_fcm, FCM_Reference):
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
    laplacian2 = nx.laplacian_spectrum(FCM_Reference.to_undirected())
    k = min(select_k(laplacian1), select_k(laplacian2))
    return sum((laplacian1[:k] - laplacian2[:k]) ** 2)


def cluster(file_location, aggregation_technique, clustering_method, n_clusters,
            f_type="tanh", infer_rule="k"):
    """
    Cluster individual FCMs by structural or dynamic similarity.

    Parameters
    ----------
    file_location : str
        Path to Excel file with one participant adjacency matrix per sheet.
    aggregation_technique : str
        How to build the reference FCM: 'AI', 'AX', 'O', or 'Z'.
    clustering_method : str
        'S' for structural (spectral) similarity, 'D' for dynamic similarity.
    n_clusters : int
    f_type : str
        Squashing function for dynamic clustering: 'sig', 'tanh', 'bivalent', 'trivalent'.
    infer_rule : str
        Inference rule for dynamic clustering: 'k', 'mk', or 'r'.

    Returns
    -------
    clusters : dict  {cluster_id: [similarity_values]}
    assignments : list of (participant_id, cluster_label) tuples
    """
    workbook = xlrd.open_workbook(file_location)
    sheet = workbook.sheet_by_index(0)
    n_concepts = sheet.nrows - 1
    n_participants = workbook.nsheets

    Allparticipants = {}
    IDs = []
    for i in range(n_participants):
        sheet = workbook.sheet_by_index(i)
        Adj_matrix = np.zeros((n_concepts, n_concepts))
        for row in range(1, n_concepts + 1):
            for col in range(1, n_concepts + 1):
                Adj_matrix[row - 1, col - 1] = sheet.cell_value(row, col)
        IDs.append(sheet.cell_value(0, 0))
        Allparticipants[sheet.cell_value(0, 0)] = Adj_matrix

    class Agent:
        def __init__(self, ID):
            self.ID = ID
            self.FCM = nx.DiGraph(Allparticipants[ID])

    def dynamic(agent, FCM_Reference_arr):
        M = 0
        W = []
        SState = _infer_steady(Allparticipants[agent.ID], n_concepts, f_type=f_type, infer_rule=infer_rule)
        SState_ref = _infer_steady(FCM_Reference_arr, n_concepts, f_type=f_type, infer_rule=infer_rule)
        iteration = 0
        for _ in range(10):
            for _ in range(100):
                rand = random.randint(1, n_concepts)
                com = random.sample(list(agent.FCM.nodes()), rand)
                my_random = {rC: random.random() * random.choice([-1, 1]) for rC in com}
                iteration += 1
                ScenarioState = _infer_scenario(com, my_random, Allparticipants[agent.ID], n_concepts,
                                                f_type=f_type, infer_rule=infer_rule)
                ScenarioState_ref = _infer_scenario(com, my_random, FCM_Reference_arr, n_concepts,
                                                    f_type=f_type, infer_rule=infer_rule)
                Change = ScenarioState - SState
                Change_ref = ScenarioState_ref - SState_ref
                M += sum((Change - Change_ref) ** 2)
            M = math.sqrt(M) / iteration
            W.append(M)
        return np.mean(W)

    def make_reference(How):
        if How == "AI":
            adj = np.zeros((n_concepts, n_concepts))
            for ag in agents:
                adj += nx.to_numpy_array(ag.FCM)
            return adj / n_participants
        if How == "AX":
            adj = np.zeros((n_concepts, n_concepts))
            count = np.zeros((n_concepts, n_concepts))
            adj_ag = np.zeros((n_concepts, n_concepts))
            for ag in agents:
                Adj_matrix = nx.to_numpy_array(ag.FCM)
                for i in range(n_concepts):
                    for j in range(n_concepts):
                        if Adj_matrix[i, j] != 0:
                            count[i, j] += 1
                adj += Adj_matrix
                adj_copy = np.copy(adj)
                for i in range(n_concepts):
                    for j in range(n_concepts):
                        adj_ag[i, j] = 0 if count[i, j] == 0 else adj_copy[i, j] / count[i, j]
            return adj_ag
        if How == "O":
            return np.ones((n_concepts, n_concepts))
        if How == "Z":
            return np.zeros((n_concepts, n_concepts))

    agents = [Agent(ID=Id) for Id in IDs]
    FCM_Reference = make_reference(aggregation_technique)

    simil = {}
    if clustering_method == "D":
        for agent in agents:
            simil[agent.ID] = dynamic(agent, FCM_Reference)
    if clustering_method == "S":
        for agent in agents:
            simil[agent.ID] = _similarity(agent.FCM, nx.DiGraph(FCM_Reference))

    X = np.array(list(simil.values()))
    km = KMeans(n_clusters=n_clusters)
    km.fit(X.reshape(-1, 1))
    assignments = list(zip(list(simil.keys()), km.labels_))

    clusters = {i: [] for i in range(n_clusters)}
    for participant_id, label in assignments:
        print(participant_id, "is in cluster {}".format(label))
        clusters[label].append(simil[participant_id])

    plt.figure(figsize=(10, 3))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=0)
    for cl in range(n_clusters):
        plt.plot(clusters[cl], np.zeros_like(clusters[cl]), 'x', markersize=8, label=cl)
    plt.legend()
    plt.savefig('Clusters.pdf')
    plt.show()

    return clusters, assignments

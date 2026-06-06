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
from statistics import mean, median


def aggregate(file_location, aggregation_technique):
    """
    Aggregate individual FCMs from an Excel workbook into a single reference FCM.

    Parameters
    ----------
    file_location : str
        Path to the Excel file containing individual adjacency matrices (one per sheet).
    aggregation_technique : str
        'AMX' – arithmetic mean excluding zeros
        'AMI' – arithmetic mean including zeros
        'MED' – median
        'GM'  – geometric mean (non-zero values only)

    Returns
    -------
    Adj_aggregated_FCM : np.ndarray
    last_sheet : xlrd.Sheet  (used for node labels)
    n_concepts : int
    """
    workbook = xlrd.open_workbook(file_location)
    sheet = workbook.sheet_by_index(0)
    n_concepts = sheet.nrows - 1

    adj = np.zeros((n_concepts, n_concepts))
    count = np.zeros((n_concepts, n_concepts))
    adj_ag = np.zeros((n_concepts, n_concepts))
    All_ADJs = []

    for i in range(workbook.nsheets):
        sheet = workbook.sheet_by_index(i)
        Adj_matrix = np.zeros((n_concepts, n_concepts))
        for row in range(1, n_concepts + 1):
            for col in range(1, n_concepts + 1):
                val = sheet.cell_value(row, col)
                Adj_matrix[row - 1, col - 1] = val
                if val != 0:
                    count[row - 1, col - 1] += 1
        All_ADJs.append(Adj_matrix)
        adj += Adj_matrix

    adj_copy = np.copy(adj)

    if aggregation_technique == "AMX":
        for i in range(n_concepts):
            for j in range(n_concepts):
                adj_ag[i, j] = 0 if count[i, j] == 0 else adj_copy[i, j] / count[i, j]

    elif aggregation_technique == "AMI":
        for i in range(n_concepts):
            for j in range(n_concepts):
                a = [ind[i, j] for ind in All_ADJs]
                adj_ag[i, j] = mean(a)

    elif aggregation_technique == "MED":
        for i in range(n_concepts):
            for j in range(n_concepts):
                a = [ind[i, j] for ind in All_ADJs]
                adj_ag[i, j] = median(a)

    elif aggregation_technique == "GM":
        import scipy.stats.mstats
        for i in range(n_concepts):
            for j in range(n_concepts):
                a = [ind[i, j] for ind in All_ADJs if ind[i, j] != 0]
                adj_ag[i, j] = float(scipy.stats.mstats.gmean(np.array(a)))

    return adj_ag, sheet, n_concepts


def visualize_aggregated(Adj_aggregated_FCM, sheet, save_edgelist=True):
    """
    Draw the aggregated FCM and optionally save its edge list.

    Parameters
    ----------
    Adj_aggregated_FCM : np.ndarray
    sheet : xlrd.Sheet  (last sheet from aggregate(), used for node labels)
    save_edgelist : bool

    Returns
    -------
    G : nx.DiGraph
    """
    G = nx.DiGraph(Adj_aggregated_FCM)

    plt.figure(figsize=(10, 10))

    everylarge = [(u, v) for (u, v, d) in G.edges(data=True) if abs(d['weight']) >= 0.75]
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if 0.5 < abs(d['weight']) < 0.75]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if 0.25 < abs(d['weight']) <= 0.5]
    everysmall = [(u, v) for (u, v, d) in G.edges(data=True) if abs(d['weight']) <= 0.25]

    label = {nod: sheet.cell_value(nod + 1, 0) for nod in G.nodes()}
    pos = nx.spring_layout(G, dim=2, k=0.75)

    nx.draw_networkx(G, pos, labels=label, font_size=7, node_size=200, node_color='lightgreen', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=everylarge, width=2, alpha=0.5, edge_color='gold')
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, alpha=0.5, edge_color='g', style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=0.5, alpha=0.5, edge_color='lightcoral', style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=everysmall, width=0.25, alpha=0.5, edge_color='lightgray', style='dashed')

    plt.show()

    if save_edgelist:
        nx.write_edgelist(G, "aggregated_edg.csv")

    return G

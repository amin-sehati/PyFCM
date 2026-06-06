# -*- coding: utf-8 -*-
"""Excel workbook readers for PyFCM adjacency matrices."""

import numpy as np
import xlrd


def _read_sheet_matrix(sheet, n_concepts, noise_threshold=None):
    adj_matrix = np.zeros((n_concepts, n_concepts))

    for row in range(1, n_concepts + 1):
        for col in range(1, n_concepts + 1):
            weight = sheet.cell_value(row, col)
            if noise_threshold is not None and abs(weight) <= noise_threshold:
                weight = 0
            adj_matrix[row - 1, col - 1] = weight

    return adj_matrix


def concept_metadata(sheet, n_concepts, principles):
    concepts = [sheet.cell_value(0, i) for i in range(1, n_concepts + 1)]
    node_name = {nod: sheet.cell_value(nod + 1, 0) for nod in range(n_concepts)}
    principle_indices = [nod for nod, name in node_name.items() if name in principles]
    return concepts, node_name, principle_indices


def read_single_fcm(file_location, noise_threshold=None):
    workbook = xlrd.open_workbook(file_location)
    sheet = workbook.sheet_by_index(0)
    n_concepts = sheet.nrows - 1
    adj_matrix = _read_sheet_matrix(sheet, n_concepts, noise_threshold)
    return adj_matrix, sheet, n_concepts


def read_participant_fcms(file_location):
    workbook = xlrd.open_workbook(file_location)
    sheet = workbook.sheet_by_index(0)
    n_concepts = sheet.nrows - 1
    participant_ids = []
    participants = {}
    matrices = []

    for i in range(workbook.nsheets):
        sheet = workbook.sheet_by_index(i)
        adj_matrix = _read_sheet_matrix(sheet, n_concepts)
        participant_id = sheet.cell_value(0, 0)
        participant_ids.append(participant_id)
        participants[participant_id] = adj_matrix
        matrices.append(adj_matrix)

    return participants, participant_ids, matrices, sheet, n_concepts

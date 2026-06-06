# -*- coding: utf-8 -*-
"""Shared FCM simulation helpers."""

import numpy as np


def make_initial_state(initial_state, n):
    if initial_state is None:
        return np.ones(n)
    if np.isscalar(initial_state):
        return np.full(n, initial_state, dtype=float)

    init_vec = np.asarray(initial_state, dtype=float)
    if len(init_vec) != n:
        raise ValueError("initial_state must be a scalar or have one value per concept")
    return init_vec


def transform_func(x, n, function_type, lambda_=1):
    x = np.asarray(x)[:n]
    if function_type == "sig":
        return 1 / (1 + np.exp(-lambda_ * x))
    if function_type == "tanh":
        return np.tanh(lambda_ * x)
    if function_type in ("bivalent", "biv"):
        return (x > 0).astype(float)
    if function_type in ("trivalent", "triv"):
        return np.sign(x).astype(float)


def infer_steady(adj_matrix, n, init_vec=None, function_type="tanh",
                 infer_rule="k", lambda_=1, tolerance=0.00001):
    init_vec = make_initial_state(init_vec, n)
    act_vec_old = init_vec.copy()
    adj_matrix_t = adj_matrix.T
    resid = 1
    ones = np.ones(n) if infer_rule == "r" else None
    while resid > tolerance:
        if infer_rule == "k":
            x = np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "mk":
            x = act_vec_old + np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "r":
            shifted_state = 2 * act_vec_old - ones
            x = shifted_state + np.matmul(adj_matrix_t, shifted_state)
        else:
            x = np.zeros(n)
        act_vec_new = transform_func(x, n, function_type, lambda_)
        resid = max(abs(act_vec_new - act_vec_old))
        if resid < tolerance:
            break
        act_vec_old = act_vec_new
    return act_vec_new


def infer_scenario(scenario_concepts, levels, adj_matrix, n, init_vec=None,
                   function_type="tanh", infer_rule="k", lambda_=1,
                   tolerance=0.00001, zero_concepts=None):
    init_vec = make_initial_state(init_vec, n)
    if zero_concepts is None:
        zero_concepts = []
    act_vec_old = init_vec.copy()
    adj_matrix_t = adj_matrix.T
    resid = 1
    ones = np.ones(n) if infer_rule == "r" else None
    while resid > tolerance:
        if infer_rule == "k":
            x = np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "mk":
            x = act_vec_old + np.matmul(adj_matrix_t, act_vec_old)
        elif infer_rule == "r":
            shifted_state = 2 * act_vec_old - ones
            x = shifted_state + np.matmul(adj_matrix_t, shifted_state)
        else:
            x = np.zeros(n)
        act_vec_new = transform_func(x, n, function_type, lambda_)
        for z in zero_concepts:
            act_vec_new[z] = 0
        for c in scenario_concepts:
            act_vec_new[c] = levels[c]
        resid = max(abs(act_vec_new - act_vec_old))
        act_vec_old = act_vec_new
    return act_vec_new

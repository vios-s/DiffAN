import numpy as np
import pandas as pd
import os
from cdt.causality.graph import CAM
import torch
import networkx as nx
from castle.algorithms import NotearsNonlinear, MCSL, GraNDAG, CORL, RL
from score_disc.stein import SCORE
from diffan.diffan import DiffAN
from diffan.experiments.diffanm_greedy import DiffANM_greedy
from diffan.pruning import cam_pruning



def run_CAM(X):
    data = pd.DataFrame(X)
    obj = CAM(pruning=True)
    output = obj.predict(data)
    adj = nx.to_numpy_array(output)
    return adj


def run_DiffANM(X):
    num_samples, n_nodes = X.shape
    diffanm = DiffAN(n_nodes)
    pred_adj, order = diffanm.fit(X)
    return pred_adj, order

def run_DiffANM_masking(X):
    num_samples, n_nodes = X.shape
    diffanm = DiffAN(n_nodes, masking = True, residue= False)
    pred_adj, order = diffanm.fit(X)
    return pred_adj, order

def run_DiffANM_sorting(X):
    num_samples, n_nodes = X.shape
    diffanm = DiffAN(n_nodes, masking = False, residue= False)
    pred_adj, order = diffanm.fit(X)
    return pred_adj, order

def run_DiffANM_greedy(X):
    num_samples, n_nodes = X.shape
    diffanm = DiffANM_greedy(n_nodes)
    pred_adj, order = diffanm.fit(X)
    return pred_adj, order

def run_SCORE(X):
    eta_G = 0.001
    eta_H = 0.001
    cam_cutoff = 0.001
    X = torch.Tensor(X).to(torch.float64)
    adj_matrix, top_order_SCORE =  SCORE(X, eta_G, eta_H, cam_cutoff)
    return adj_matrix, top_order_SCORE

def run_NotearsMLP(X):
    n = NotearsNonlinear(device_type = "gpu")
    n.learn(X)
    return n.causal_matrix

def run_MCSL(X):
    n = MCSL(device_type = "gpu")
    n.learn(X)
    return n.causal_matrix

def run_CORL(X):
    n = CORL(device_type = "gpu")
    n.learn(X)
    return n.causal_matrix

def run_RL(X):
    n = RL(device_type = "gpu")
    n.learn(X)
    return n.causal_matrix

def run_GrandDAG(X):
    X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
    cam_cutoff = 0.001
    gnd = GraNDAG(input_dim=X.shape[1], use_pns = True,device_type = "gpu")
    gnd.learn(data=X)
    out_dag = cam_pruning(gnd.causal_matrix, X, cam_cutoff)
    return out_dag
# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
import time
import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
np.set_printoptions(precision=3)
# internal
from diffan.utils import get_value_from_str
# baselines
from cdt.metrics import SID
from castle.metrics import MetricsDAG
from diffan.experiments.baselines import run_DiffANM_greedy,run_DiffANM_masking, run_DiffANM_sorting, run_DiffANM, run_SCORE, run_GrandDAG, run_NotearsMLP, run_MCSL,run_CAM,run_RL, run_CORL

def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    dataset_folder = Path(data['dataset_folder'])
    datasets_path_list = sorted(list(dataset_folder.glob("*")))[::-1]
    
    assert len(datasets_path_list) > 0, "Zero datasets were found, please use 'gen_synth_data.py' for generating the data or check if the path is correct"
    methods = [ ("DiffANM_sorting0", run_DiffANM_sorting),("DiffANM_masking0", run_DiffANM_masking)]#("DiffANM_residue_all_diag1_5k", run_DiffANM)]#,("DiffANM_masking", run_DiffANM_masking),("DiffANM_sorting", run_DiffANM_sorting),("DiffANM_greedy", run_DiffANM_greedy)("DiffANM_const", run_DiffANM),("SCORE", run_SCORE),("CAM", run_CAM),("GranDAG", run_GrandDAG)]# ("RL", run_RL), ("CORL", run_CORL),("SCORE", run_SCORE),("GranDAG", run_GrandDAG), ("NotearsMLP", run_NotearsMLP), ("DiffANM_greedy", run_DiffANM_greedy)]#("CAM", run_CAM),  ("DiffANM_new", run_DiffANM),
    for dataset_path in datasets_path_list:
        
        exp_name = dataset_path.name

        seed = get_value_from_str(exp_name,"seed",int)
        sem = get_value_from_str(exp_name,"sem")
        nnodes = get_value_from_str(exp_name,"nnodes",int)
        
        if not (nnodes in [20] and seed in list(range(3)) and sem == "gp"):
            continue

        dataset_path  = dataset_folder / exp_name
        dataset_adj_path = dataset_path /  "adj_matrix.csv"
        train_set_path = dataset_path /  "train_set.csv"
        true_adj = pd.read_csv(dataset_adj_path).to_numpy()
        X = pd.read_csv(train_set_path).to_numpy()    
        X = X[np.random.rand(2000).argsort()]
        num_samples, n_nodes = X.shape

        # zero mean, unit variance normalization
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        
        print(f"n samples {num_samples} and n nodes {n_nodes} \n Exp {exp_name}")

        for method_name, method_fn in methods:
            exp_folder_path = exp_results_folder / "synthetic" / exp_name / method_name

            if (exp_folder_path / "metrics.csv").exists():
                print(f"{method_name} experiment for dataset {exp_name} already exists")
                continue
            exp_folder_path.mkdir(parents = True, exist_ok = True)

            print(f"Running {method_name}")
            start = time.perf_counter() 
            #try:
            pred_adj = method_fn(X)
            '''except:
                print(f"Error in {method_name} for dataset {exp_name}")
                continue'''

            stop = time.perf_counter() 
            run_time = stop-start

            
            if isinstance(pred_adj, tuple):
                pred_adj = pred_adj[0]


            mt = MetricsDAG(pred_adj, true_adj).metrics # return a dict
            mt["sid"] = SID(true_adj, pred_adj).item()
            mt["runtime"] = run_time


            print(mt)

            pd.DataFrame(pred_adj).to_csv(exp_folder_path / "adj_pred.csv", index = False)
            pd.DataFrame(true_adj).to_csv(exp_folder_path / "adj_true.csv", index = False)
            pd.DataFrame(mt, index=[0]).to_csv(exp_folder_path / "metrics.csv")


if __name__ == "__main__":
    main()

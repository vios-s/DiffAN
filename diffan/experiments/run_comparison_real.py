# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import time
import torch 
from cdt.metrics import SID
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['CASTLE_BACKEND'] = 'pytorch'
np.set_printoptions(precision=3)
# internal
# baselines
from castle.metrics import MetricsDAG
from diffan.experiments.baselines import run_DiffANM, run_SCORE, run_GrandDAG, run_NotearsMLP, run_MCSL, run_CAM, run_DiffANM_greedy



def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    dataset_folder = Path(data['real_dataset_folder'])
    datasets_path_list = sorted(list(dataset_folder.glob("*")))
    assert len(datasets_path_list) > 0, "Zero datasets were found, please use 'gen_synth_data.py' for generating the data or check if the path is correct"
    methods = [("DiffANM_residue", run_DiffANM)] #("DiffANM_greedy", run_DiffANM_greedy), ("SCORE", run_SCORE), ("GranDAG", run_GrandDAG), ("NotearsMLP", run_NotearsMLP),("MCSL", run_MCSL)] ("DiffANM_const", run_DiffANM)]#, #, ("CAM", run_CAM), ("SCORE", run_SCORE), ("GranDAG", run_GrandDAG)

    for dataset_path in datasets_path_list:
        for data_path in dataset_path.glob("data*"):
            exp_name = dataset_path.name
            #if exp_name == "sachs":#sachs,syntren
            #    continue
        
            variable = "data"
            value_init_pos = str(data_path).rfind(variable) + len(variable)
            value = int(str(data_path)[value_init_pos:value_init_pos+1])

            exp_name += f"_{value}"
            true_adj = np.load(data_path.parent / f"DAG{value}.npy")
            X = np.load(data_path)
            num_samples, n_nodes = X.shape

            print(f"n samples {num_samples} and n nodes {n_nodes} \n Exp {exp_name}")

            for method_name, method_fn in methods:
                exp_folder_path = exp_results_folder / exp_name / method_name

                '''if (exp_folder_path / "metrics.csv").exists():
                    print(f"{method_name} experiment for dataset {exp_name} already exists")
                    continue'''
                exp_folder_path.mkdir(parents = True, exist_ok = True)

                

                print(f"Running {method_name}")
                start = time.perf_counter() 
                
                pred_adj = method_fn(X)

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
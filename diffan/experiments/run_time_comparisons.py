# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
np.set_printoptions(precision=3)
# internal
from diffscm.diffanm import DiffANM
from diffscm.utils import num_errors, get_value_from_str
from score_disc.stein import SCORE

def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    dataset_folder = Path(data['dataset_folder'])
    datasets_path_list = sorted(list(dataset_folder.glob("*")))
    assert len(datasets_path_list) > 0, "Zero datasets were found, please use 'gen_synth_data.py' for generating the data or check if the path is correct"

    method_name = "DiffANM_residue_A100_scaling" #DiffANM_scaling,DiffANM_A100_scaling, SCORE_scaling

    for dataset_path in datasets_path_list:

        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors", "sample_size")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))

        exp_name = dataset_path.name
        # Filter Experiments
        seed = get_value_from_str(exp_name, "seed", int)
        nnodes = get_value_from_str(exp_name, "nnodes", int)
        nedges = get_value_from_str(exp_name, "nedges", int)
        sem = get_value_from_str(exp_name, "sem")

        if not (seed in list(range(3)) and nnodes in [500] and nedges == 5 and sem == "mlp"):
            continue
        
        dataset_adj_path = dataset_path /  "adj_matrix.csv"
        train_set_path = dataset_path /  "train_set.csv"
        true_adj = pd.read_csv(dataset_adj_path).to_numpy()
        X = pd.read_csv(train_set_path).to_numpy()
        num_samples, n_nodes = X.shape
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = torch.FloatTensor(X).to(device)
        print(f"n samples {num_samples} and n nodes {n_nodes} \n Exp {exp_name}")
        exp_folder_path = exp_results_folder / method_name / exp_name

        if (exp_folder_path / "5k_results.csv").exists():
                print(f"{method_name} experiment for dataset {exp_name} already exists")
                continue

        for data_size_ratio in [0.05,0.1,0.5,1.0]:#0.001,0.01,0.02,0.03,0.04,
            n_nodes = X.shape[1]
            num_samples = round(data_size_ratio * X.shape[0])            
            X_subsampled = X[np.random.rand(num_samples).argsort()]
            print(f"n samples {num_samples} and n nodes {n_nodes} \n Exp {exp_name}")


            if "DiffANM" in method_name:
                diffanm = DiffANM(n_nodes, batch_size = 512)
                
                start = time.perf_counter() 
                
                diffanm.train_score(X_subsampled)
                order = diffanm.topological_ordering(X_subsampled, step = None, eval_batch_size = 8)

                stop = time.perf_counter() 
                run_time = stop-start
                order_errors = num_errors(order, true_adj)


                results_dict["dataset_name"].append(exp_name)
                results_dict["ordering_errors"].append(order_errors)
                results_dict["sample_size"].append(num_samples)
                results_dict["runtime"].append(run_time)
                results_dict["method"].append("DiffANM")
                
                print(f"{exp_name} - DiffANM - num_samples {num_samples}- Time {run_time} - Order errors {order_errors}")

            else:
                eta_G = 0.001
                eta_H = 0.001
                cam_cutoff = 0.001
                
                        
                start = time.perf_counter() 
                
                order =  SCORE(X_subsampled.to(torch.float64).to('cpu'), eta_G, eta_H, cam_cutoff, pruning = None)

                stop = time.perf_counter() 
                run_time = stop-start
                order_errors = num_errors(order, true_adj)

                results_dict["dataset_name"].append(exp_name)
                results_dict["ordering_errors"].append(order_errors)
                results_dict["sample_size"].append(num_samples)
                results_dict["runtime"].append(run_time)
                results_dict["method"].append("SCORE")

                print(f"{exp_name} - SCORE - num_samples {num_samples}- Time {run_time} - Order errors {order_errors}")

        exp_folder_path.mkdir(parents = True, exist_ok = True)
        pd.DataFrame(results_dict).to_csv(exp_folder_path / "5k_results.csv")




if __name__ == "__main__":

    main()

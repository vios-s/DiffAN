# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
import scipy
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
np.set_printoptions(precision=3)
# internal
from diffscm.diffanm import DiffANM
from diffscm.utils import num_errors, get_value_from_str, full_DAG, fullAdj2Order
from castle.metrics import MetricsDAG

def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    dataset_folder = Path(data['dataset_folder'])
    datasets_path_list = sorted(list(dataset_folder.glob("*")))
    assert len(datasets_path_list) > 0, "Zero datasets were found, please use 'gen_synth_data.py' for generating the data or check if the path is correct"

    for dataset_path in datasets_path_list[::-1]:

        exp_name = dataset_path.name
        # Filter Experiments
        seed = get_value_from_str(exp_name, "seed", int)
        nnodes = get_value_from_str(exp_name, "nnodes", int)
        nedges = get_value_from_str(exp_name, "nedges", int)
        sem = get_value_from_str(exp_name, "sem")

        if not (seed in list(range(3)) and nnodes in [20] and nedges == 5 and sem == "gp"):
            continue
        try:
            dataset_adj_path = dataset_path /  "adj_matrix.csv"
            train_set_path = dataset_path /  "train_set.csv"
            true_adj = pd.read_csv(dataset_adj_path).to_numpy()
            X = pd.read_csv(train_set_path).to_numpy()
        except:
            continue
        num_samples, n_nodes = X.shape
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = torch.FloatTensor(X).to(device)
        
        for overfit in [True, False]:
            results_dict_keys = ("dataset_name", "method", "runtime", "order_errors", "order_errors_jac", "order_errors_oneshot")
            results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
            exp_folder_path = exp_results_folder / "DiffANM_overfit_ablations" / exp_name / str(overfit)
            '''if (exp_folder_path / "results.csv").exists():
                print(f"{exp_name} already exists")
                continue'''
            
            if overfit:
                fixed = int(2e3)
            else:
                fixed = None

            diffanm = DiffANM(n_nodes)
            
            start = time.perf_counter() 
            
            diffanm.train_score(X, fixed=fixed)
            order, sorteds, jacobian_vars = diffanm.topological_ordering(X, step = 5)
            stop = time.perf_counter() 
            run_time = stop-start
            jac = jacobian_vars[0]
            jac_norm_by_diag = (jac / (jac * np.eye(jac.shape[0]) @ np.ones_like(jac))) - np.eye(jac.shape[0])
            jac_sort = np.argsort(jac_norm_by_diag.sum(1))

            order_errors = num_errors(order, true_adj)
            order_errors_jac = num_errors(jac_sort, true_adj)
            order_errors_oneshot = num_errors(sorteds[0][0][::-1], true_adj)

            results_dict["dataset_name"].append(exp_name)
            results_dict["order_errors"].append(order_errors)
            results_dict["order_errors_jac"].append(order_errors_jac)
            results_dict["order_errors_oneshot"].append(order_errors_oneshot)
            results_dict["runtime"].append(run_time)
            results_dict["method"].append(overfit)
            
            print(f"{exp_name} - Order errors {order_errors} - Time {run_time} ")

            exp_folder_path.mkdir(parents = True, exist_ok = True)
            pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")




if __name__ == "__main__":

    main()

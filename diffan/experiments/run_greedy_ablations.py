# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.set_printoptions(precision=3)
# internal
from diffscm.diffanm import DiffANM
from diffscm.experiments.diffanm_greedy import DiffANM_greedy
from diffscm.utils import num_errors, get_value_from_str


def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    dataset_folder = Path(data['dataset_folder'])
    datasets_path_list = sorted(list(dataset_folder.glob("*")))
    assert len(datasets_path_list) > 0, "Zero datasets were found, please use 'gen_synth_data.py' for generating the data or check if the path is correct"

    for dataset_path in datasets_path_list:

        

        exp_name = dataset_path.name
        # Filter Experiments
        seed = get_value_from_str(exp_name, "seed", int)
        nnodes = get_value_from_str(exp_name, "nnodes", int)
        nedges = get_value_from_str(exp_name, "nedges", int)
        sem = get_value_from_str(exp_name, "sem")

        #if not (seed in list(range(3)) and nnodes in [20] and nedges == 5 and sem == "gp"):
        #    continue
        if not (seed in list(range(3)) and nnodes in [500]):
            continue
        try:
            dataset_adj_path = dataset_path /  "adj_matrix.csv"
            train_set_path = dataset_path /  "train_set.csv"
            true_adj = pd.read_csv(dataset_adj_path).to_numpy()
            X = pd.read_csv(train_set_path).to_numpy()
            X = X[np.random.rand(3000).argsort()]
        except:
            continue
        num_samples, n_nodes = X.shape
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = torch.FloatTensor(X).to(device)

        print(f"n samples {num_samples} and n nodes {n_nodes} \n Exp {exp_name}")

        method_name = "opt" # greedy,opt

        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
        exp_folder_path = exp_results_folder / "DiffANM_greedy_ablations" / method_name / exp_name


        diffanm = DiffANM(n_nodes, dropout_type = "const")
        
        start_train = time.perf_counter() 
        diffanm.train_score(X)
        stop_train = time.perf_counter() 
        run_time_train = stop_train-start_train

        start_inf = time.perf_counter()
        order = diffanm.topological_ordering(X, eval_batch_size = 64)
        stop_inf = time.perf_counter() 
        run_time_inf = stop_inf-start_inf
        run_time = run_time_train + run_time_inf
        
        order_errors = num_errors(order, true_adj)


        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["runtime"].append(run_time)
        results_dict["method"].append(method_name)
        
        print(f"{exp_name} - DiffANM {method_name} - Order errors {order_errors} - Time {run_time} ")
        
        method_name = "oneshot_5"

        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
        exp_folder_path = exp_results_folder / "DiffANM_greedy_ablations" / method_name / exp_name

        start_inf = time.perf_counter()
        order, sorteds, jacobian_vars = diffanm.topological_ordering(X,step = 5, eval_batch_size = 64)
        stop_inf = time.perf_counter() 
        run_time_inf = stop_inf-start_inf
        run_time = run_time_train + run_time_inf
        
        order_errors = num_errors(sorteds[0][0][::-1], true_adj)


        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["runtime"].append(run_time)
        results_dict["method"].append(method_name)
        
        print(f"{exp_name} - DiffANM {method_name} - Order errors {order_errors} - Time {run_time} ")
        

        exp_folder_path.mkdir(parents = True, exist_ok = True)
        pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")

        method_name = "oneshot_25"

        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
        exp_folder_path = exp_results_folder / "DiffANM_greedy_ablations" / method_name / exp_name

        start_inf = time.perf_counter()
        order, sorteds, jacobian_vars = diffanm.topological_ordering(X,step = 25, eval_batch_size = 64)
        stop_inf = time.perf_counter() 
        run_time_inf = stop_inf-start_inf
        run_time = run_time_train + run_time_inf
        
        order_errors = num_errors(sorteds[0][0][::-1], true_adj)


        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["runtime"].append(run_time)
        results_dict["method"].append(method_name)
        
        print(f"{exp_name} - DiffANM {method_name} - Order errors {order_errors} - Time {run_time} ")
        

        exp_folder_path.mkdir(parents = True, exist_ok = True)
        pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")

        method_name = "oneshot_50"

        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
        exp_folder_path = exp_results_folder / "DiffANM_greedy_ablations" / method_name / exp_name

        start_inf = time.perf_counter()
        order, sorteds, jacobian_vars = diffanm.topological_ordering(X, step = 50, eval_batch_size = 64)
        stop_inf = time.perf_counter() 
        run_time_inf = stop_inf-start_inf
        run_time = run_time_train + run_time_inf
        
        order_errors = num_errors(sorteds[0][0][::-1], true_adj)


        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["runtime"].append(run_time)
        results_dict["method"].append(method_name)
        
        print(f"{exp_name} - DiffANM {method_name} - Order errors {order_errors} - Time {run_time} ")
        

        exp_folder_path.mkdir(parents = True, exist_ok = True)
        pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")
        
        '''method_name = "greedy" # greedy,opt

        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
        exp_folder_path = exp_results_folder / "DiffANM_greedy_ablations" / method_name / exp_name

        diffanm_greedy = DiffANM_greedy(n_nodes)
        
        start = time.perf_counter() 
        
        order = diffanm_greedy.topological_ordering(X)

        stop = time.perf_counter() 
        run_time = stop-start
        order_errors = num_errors(order, true_adj)


        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["runtime"].append(run_time)
        results_dict["method"].append(method_name)
        
        print(f"{exp_name} - DiffANM {method_name} - Order errors {order_errors} - Time {run_time} ")

        exp_folder_path.mkdir(parents = True, exist_ok = True)
        pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")'''


        




if __name__ == "__main__":

    main()

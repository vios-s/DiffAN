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

        if not (seed in list(range(1)) and nnodes in [20] and nedges == 5 and sem == "gp"):
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
        print(exp_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = torch.FloatTensor(X).to(device)
        results_dict_keys = ("dataset_name", "method", "runtime", "ordering_errors")
        results_dict = dict(zip(results_dict_keys, [ [] for _ in range(len(results_dict_keys))]))
        exp_folder_path = exp_results_folder / "DiffANM_jacobian_ablations" / exp_name
        if (exp_folder_path / "results.csv").exists():
            print(f"{exp_name} already exists")
            continue


        diffanm = DiffANM(n_nodes, dropout_type = "const")
        
        start = time.perf_counter() 
        
        diffanm.train_score(X)
        
        #order = diffanm.topological_ordering_correction(X, step = 10)
        #order = diffanm.topological_ordering_correction(X, step = None, use_residue = False)
        order_mask = diffanm.topological_ordering(X, step = None)
        
        print(f" without {num_errors(order_mask, true_adj)}")#With residue {num_errors(order, true_adj)},
        '''
        order = diffanm.topological_ordering_correction(X, step = 5)
        order_mask, sorteds, jacobian_vars = diffanm.topological_ordering(X, step = 5)
        print(f"new ordering {num_errors(order, true_adj)}, old {num_errors(order_mask, true_adj)}")
        order = diffanm.topological_ordering_correction(X, step = 10)
        order_mask, sorteds, jacobian_vars = diffanm.topological_ordering(X, step = 10)
        print(f"new ordering {num_errors(order, true_adj)}, old {num_errors(order_mask, true_adj)}")
        order = diffanm.topological_ordering_correction(X, step = 30)
        order_mask, sorteds, jacobian_vars = diffanm.topological_ordering(X, step = 30)
        print(f"new ordering {num_errors(order, true_adj)}, old {num_errors(order_mask, true_adj)}")
        
        stop = time.perf_counter()
        run_time = stop-start
        jac = jacobian_vars[0]
        #pred_adj = (jac / (jac *np.eye(jac.shape[0]) @ np.ones_like(jac))>1).astype(int).T
        diag_to_line = (jac * np.eye(jac.shape[0])) @ np.ones_like(jac)
        diag_to_col = np.ones_like(jac) @ (jac * np.eye(jac.shape[0]))
        jac_norm_by_diag = (jac / diag_to_line)
        jac_norm = jac / diag_to_line * diag_to_col
        jac_norm_sort = np.argsort(jac_norm.sum(1))

        jac_sort = np.argsort(jac_norm_by_diag.sum(1))
        get_jac_norm = lambda j: (j / (j * np.eye(j.shape[0]) @ np.ones_like(j)))

        get_adj = lambda th : (jac_norm_by_diag > th).astype(int).T
        metrics = [MetricsDAG(get_adj(th/100), true_adj).metrics["shd"] for th in range(100)]
        
        get_dagness = lambda adj : np.trace(scipy.linalg.expm(adj)) - n_nodes
        ordering_dag = full_DAG(sorteds[0][0][::-1]) 
        import networkx as nx
        G = nx.from_numpy_matrix(true_adj, create_using=nx.DiGraph); true_order = list(nx.topological_sort(G))

        metrics = [MetricsDAG(get_adj(th/100), true_adj).metrics["shd"] for th in range(100)]
        metrics_masked = [MetricsDAG(get_adj(th/100)*full_DAG(order), true_adj).metrics["shd"] for th in range(100)]
        metrics_dagness = [get_dagness(get_adj(th/100)) for th in range(100)]
        out_dag = diffanm.pruning(order, X.detach().cpu().numpy())
        #torch.trace(torch.matrix_exp(A)) - self.num_nodes
        MetricsDAG(out_dag, true_adj).metrics["shd"]

        a = [[*sorteds[i][0][::-1],*order[-i:]] for i in range(1,len(sorteds))]

        order_errors = num_errors(order, true_adj), num_errors(sorteds[0][0][::-1], true_adj), num_errors(jac_sort, true_adj)
        order_errors_oneshot = num_errors(sorteds[0][0][::-1], true_adj)

        results_dict["dataset_name"].append(exp_name)
        results_dict["ordering_errors"].append(order_errors)
        results_dict["runtime"].append(run_time)
        results_dict["method"].append("DiffANM")
        
        print(f"{exp_name} - Order errors {order_errors} - Time {run_time} ")

        exp_folder_path.mkdir(parents = True, exist_ok = True)
        pd.DataFrame(results_dict).to_csv(exp_folder_path / "results.csv")'''




if __name__ == "__main__":

    main()

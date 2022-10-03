
from diffscm.experiments.datasets.simulator import IIDSimulation, DAG
import pandas as pd
from pathlib import Path
import json
from typing import Tuple
import numpy as np
with open(Path(__file__).parent.parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)

dataset_folder = data['dataset_folder']
n_nodes = [20,50]#[10,20,50,100,200]
n_seeds = 3
n_edges = [5,1]
train_set_size = 3000
num_samples = train_set_size 
graph_types = ["ER","SF"]
sem_types = ["gp"]#,"gp"
noise_types = ["exp","gauss","laplace"]
noise_scales = [(0.4,0.8),(0.8,1.2)]#(1,)
for seed in range(n_seeds):
    for sem_type in sem_types:
        for n_node in n_nodes:
            for n_edge in n_edges:
                for graph_type in graph_types:
                    for noise_type in noise_types:
                        for noise_scale in noise_scales:
                    
                            noise_scales_str = "-".join([str(i) for i in noise_scale])
                            dataset_string = f"nnodes{n_node}_nedges{n_edge}_graph{graph_type}_sem{sem_type}_noisetype{noise_type}_noisesc{noise_scales_str}_seed{seed}"
                            dataset_folder_path = Path(dataset_folder) / dataset_string
                            dataset_folder_path.mkdir(exist_ok = True)

                            print(f"Creating dataset {dataset_string}")
                            if graph_type == "ER":
                                weighted_random_dag = DAG.erdos_renyi(n_nodes=n_node, n_edges=n_edge*n_node, seed=seed)
                            elif graph_type == "SF":
                                weighted_random_dag = DAG.scale_free(n_nodes=n_node, n_edges=n_edge*n_node, seed=seed)
                            else:
                                raise Exception(f"graph type {graph_type} is not available")
                            print("Sampling...")
                            if isinstance(noise_scale, Tuple):
                                noise_scale = np.random.uniform(noise_scale[0], noise_scale[1], size = n_node) 

                            dataset = IIDSimulation(W=weighted_random_dag, n=num_samples, 
                                        method='nonlinear', sem_type=sem_type,
                                        noise_scale = noise_scale, noise_type = noise_type)

                            true_causal_matrix, X = dataset.B, dataset.X
                            pd.DataFrame(true_causal_matrix).to_csv(dataset_folder_path / "adj_matrix.csv", index = False)
                            pd.DataFrame(X[:train_set_size]).to_csv(dataset_folder_path / "train_set.csv", index = False)


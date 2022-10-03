# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
from cdt.data import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CASTLE_BACKEND'] = 'pytorch'
np.set_printoptions(precision=3)
# internal
from diffan.diffanm import DiffANM
# baselines
from diffan.experiments.baselines import run_DiffANM

def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    results = dict()
    
    all_pairs, labels = load_dataset('tuebingen')

    methods = [("DiffANM_pair", run_DiffANM)]
    for i, data_pair in enumerate(all_pairs.iterrows()):
        pair_name, data = data_pair
        X = np.stack(data.to_numpy(), axis = 1)
        num_samples, n_nodes = X.shape

        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        
        
        diffanm = DiffANM(n_nodes)
        X = torch.FloatTensor(X).to(diffanm.device)
        diffanm.train_score(X)
        order = diffanm.topological_ordering(X)
        print(order)
        results[pair_name] = order

    exp_folder_path = exp_results_folder / "DiffANM_pairs"
    exp_folder_path.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(results).to_csv(exp_folder_path /  "results.csv", index = False)


if __name__ == "__main__":
    main()
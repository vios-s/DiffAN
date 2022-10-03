# external
import numpy as np
from pathlib import Path
import pandas as pd
import json
import torch 
from cdt.data import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['CASTLE_BACKEND'] = 'pytorch'
np.set_printoptions(precision=3)

# internal
# baselines
from cdt.causality.pairwise import ANM,BivariateFit,CDS,GNN,IGCI,RECI

def main():

    with open(Path(__file__).parent.absolute() / 'config.json', 'r') as f:
        data = json.load(f)
    exp_results_folder = Path(data['exp_results_folder'])
    results = dict()
    
    all_pairs, labels = load_dataset('tuebingen')
    data = all_pairs.iloc[:10]
    
    obj = ANM()
    output = obj.predict(data)
    print(f" ANM: {output}")

    obj = BivariateFit()
    output = obj.predict(data)
    print(f" BivariateFit: {output}")

    obj = CDS()
    output = obj.predict(data)
    print(f" CDS: {output}")

    '''obj = GNN()
    output = obj.predict(data)
    print(f" GNN: {output}")'''

    obj = IGCI()
    output = obj.predict(data)
    print(f" IGCI: {output}")

    obj = RECI()
    output = obj.predict(data)
    print(f" RECI: {output}")


    #exp_folder_path = exp_results_folder / "baseline_pairs"
    #exp_folder_path.mkdir(parents = True, exist_ok = True)
    #pd.DataFrame(results).to_csv(exp_folder_path /  "results.csv", index = False)


if __name__ == "__main__":
    main()
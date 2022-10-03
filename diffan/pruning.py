import numpy as np
import torch
import pandas as pd
import os
from pathlib import Path
from cdt.utils.R import launch_R_script
from diffan.utils import np_to_csv
import tempfile

def cam_pruning(A, X, cutoff):
    #save_path = "."
    with tempfile.TemporaryDirectory() as save_path:
        pruning_path = Path(__file__).parent / "pruning_R_files/cam_pruning.R"

        data_np = X #np.array(X.detach().cpu().numpy())
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(A, save_path)

        arguments = dict()
        arguments['{PATH_DATA}'] = data_csv_path
        arguments['{PATH_DAG}'] = dag_csv_path
        arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
        arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
        arguments['{CUTOFF}'] = str(cutoff)
        arguments['{VERBOSE}'] = "FALSE" # TRUE, FALSE


        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        
        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)
    return dag




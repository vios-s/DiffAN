
import numpy as np
import uuid
import pandas as pd
import os 

def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err
    
def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    #output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
    output = os.path.join(save_path, 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def get_value_from_str(exp_name : str, variable : str, type_func = str):
    value_init_pos = exp_name.rfind(variable) + len(variable)
    if exp_name.rfind(variable) == -1:
        return np.nan 
    new_str = exp_name[value_init_pos:]
    end_pos = new_str.find("_") 
    if end_pos == -1:
        return type_func(new_str)
    else:
        return type_func(new_str[:end_pos])
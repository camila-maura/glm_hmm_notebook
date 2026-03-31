# Imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
import pynapple as nap

# Utils
def load_session_fold_lookup(file_path):
    '''By Ashwood
    '''
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table
session_fold_table = load_session_fold_lookup("IBL/CSHL_008_session_fold_lookup.npz")

def load_data(animal_file):
    '''By Ashwood
    '''
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


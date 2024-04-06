## Import packages ##
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import argparse
import math

from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import LabelEncoder

import io

## Define parameters ##
######################################################################
data_path = "mushrooms.csv" # specify location of mushroom.csv
input_dim = 22              # equal to number of features describing each mushroom
hidden_dim = 90             # number of hidden neurons
output_dim = 1              # number of output neurons
device = 'cpu'              # we will be using CPU in this practical
batch_size = 256            # specify batch size
######################################################################
######################################################################

## Define model evaluation function ##
######################################################################
def evaluate(model, loader): # evaluates the trained model

    # we need to switch the model into the evaluation mode
    model.eval()

    # create a list to store the prediction results
    res_store = []
    for batch in loader:
        x, y = batch

        # make a prediction for a data sample "x"
        pred = model(x)
        pred = (pred > 0.5).float().squeeze(1)
        y = y.squeeze(1)

        # if the prediction is correct, append True; else append False
        res_store += (pred == y).tolist()

    # return the classification accuracy
    acc = sum(res_store)/len(res_store)
    return acc
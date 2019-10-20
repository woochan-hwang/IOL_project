import os
import numpy as np
import pandas as pd
from pandas import ExcelFile
import torch
import torch.nn as nn
from dataLoader import DataLoader, IOLDataset


df = pd.read_excel(str(os.getcwd()) + '/IOL_clean_data.xlsx')
IOLDataset = IOLDataset(df, NaN = 999, max_length = 5, verbose = False)
dataset = IOLDataset.create_dataset(bishop_included = False) # v high proportion of NaN

print(type(dataset), np.shape(dataset))

class SimpleClassifier:
    def __init__(self, dataset):
        self.dataset = dataset

    def set_output_size(self):
        self.set_output_size = np.size(self.dataset['outcome'])[0]

    def model(self):
        return

    def run(self):
        return

    def save(self):
        return

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self, data, batch_size, shuffle):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.prepare()

    def prepare(self):

        for i in range(0, self.n_samples, self.batch_size):
            

        
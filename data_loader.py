import torch
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self, data, batch_size = 2, shuffle = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = data.shape[0]
        self.prepare()

    def prepare(self):
        data_array = []
        for i in range(0, self.n_samples, self.batch_size):
            data_array.append(self.data[i:i+self.batch_size])
        data_array = np.array(data_array)
        print(data_array.shape)


x = np.random.rand(10).reshape(10, 1)
d = DataLoader(x)

        
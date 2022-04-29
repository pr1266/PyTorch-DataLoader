import torch
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self, data, batch_size = 16, shuffle = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = data.shape[0]
        self.prepare()

    def prepare(self):
        data_array = []        
        for i in range(0, self.n_samples, self.batch_size):            
            step = self.batch_size
            if i + self.batch_size > self.n_samples:
                step = self.n_samples - i
            print(step)
            data_array.append(self.data[i:i+step])
        data_array = np.array(data_array, dtype = object)
        print(data_array.shape)


x = np.random.rand(100, 32, 32)
d = DataLoader(x)

        
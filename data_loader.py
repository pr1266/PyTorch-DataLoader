import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.system('cls')

class DataLoader:

    def __init__(self, data, batch_size = 16, shuffle = True):
        self.index = 0
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = data.shape[0]
        temp = (1,)
        self.sample_shape = np.array(temp + data.shape[1:])        
        self.prepare()

    def split_to_batch(self):
        data_array = []
        for i in range(0, self.n_samples, self.batch_size):            
            step = self.batch_size
            if i + self.batch_size > self.n_samples:
                step = self.n_samples - i
            data_array.append(np.array(self.data[i:i+step], dtype = np.float32))
        data_array = np.array(data_array, dtype = object)
        return data_array

    def padding(self):
        for index, i in enumerate(self.data_array):
            if i.shape[0] != self.batch_size:
                for j in range(i.shape[0], self.batch_size):
                    padd_data = np.zeros(self.sample_shape)
                    self.data_array[index] = np.concatenate((self.data_array[index], padd_data))
                    
        new_data = [] 
        for i in self.data_array:
            new_data.append(i)

        new_data = np.array(new_data)
        self.data_array = new_data
        return self.data_array

    def prepare(self):
        
        self.data_array = self.split_to_batch()
        self.padded_data = self.padding()
        final_size = (self.padded_data.shape[0], self.batch_size) + tuple(self.sample_shape[1:])
        self.tensor_data = torch.from_numpy(self.padded_data.reshape(final_size))
        self.data_len = self.padded_data.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 0
        if self.index > self.data_len - 1:
            raise StopIteration
        return self.tensor_data[self.index]

x = np.random.rand(100, 32, 32, 3)
d = iter(DataLoader(x))

print(next(d).shape)

        
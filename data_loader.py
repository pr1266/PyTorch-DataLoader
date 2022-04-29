import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataLoader:

    def __init__(self, data, batch_size = 16, shuffle = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = data.shape[0]
        temp = (1,)
        self.sample_shape = np.array(temp + data.shape[1:])
        # temp = tuple()
        # temp.append(1)
        # for i in self.sample_shape:
        #     temp.append(i)
        # print(temp)
        print(self.sample_shape)
        self.prepare()

    def split_to_batch(self):
        data_array = []        
        for i in range(0, self.n_samples, self.batch_size):            
            step = self.batch_size
            if i + self.batch_size > self.n_samples:
                step = self.n_samples - i
            data_array.append(self.data[i:i+step])
        return data_array

    def padding(self):
        for i in self.data_array:
            if i.shape[0] != self.batch_size:
                print(i.shape)
                for j in range(i.shape[0], self.batch_size):
                    padd_data = np.zeros(self.sample_shape)                         
                    i = np.append(i, padd_data)
                print(i.shape)


    def prepare(self):
        
        self.data_array = self.split_to_batch()
        self.padded_data = self.padding()


x = np.random.rand(100, 32, 32, 3)
d = DataLoader(x)

        
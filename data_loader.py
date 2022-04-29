import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def split(self):

        x_tr
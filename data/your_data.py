

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from torchvision import datasets, transforms



class YourData(data.Dataset):
    def __init__(self, 
            train=False,
        ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.train = train

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        
        y = torch.randint(0, 2, (1,)).type(torch.float)
        x = torch.ones(1024) * y
        # x += torch.normal(0, 0.01, (1024,))
        
        return x, y
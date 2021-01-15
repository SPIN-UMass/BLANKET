
import torch.optim as optim
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from torchvision import datasets, transforms
import threading
import scipy.stats as st


import torch


def extract_time_size_to_tensor ( inp):
    t = np.array([i[0] for i in inp])
    s = np.array([i[1] for i in inp])
    t -= t[0]
    return torch.from_numpy(t).float(),torch.from_numpy(s).float()





class Generator(nn.Module):
    def __init__(self,inp,out):
        super(Generator, self).__init__()
        
        self.fc1_size   = nn.Linear(inp, 300)
        self.fc1_time   = nn.Linear(inp, 300)
        self.fc_size  = nn.Linear(300, out)
        self.fc_time   = nn.Linear(300, out)


    
    def forward(self, x):
        t,s = extract_time_size_to_tensor(x)
        print(t,s)
        out_time = F.relu(self.fc1_time(t))
        out_size = F.relu(self.fc1_size(s))

        out_size = F.relu(self.fc_size(out_size))
        out_time = F.relu(self.fc_time(out_time))
        

        return out_size,out_time
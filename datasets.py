import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
from dataset_ILC import ILCDataset
from ReadText import ReadText

def ilc_dataset(root=ReadText("Grav_ILC_setting3.txt")["Data Folder"]):
    dataset = ILCDataset(root)
    #dataset.blacklist()
    return dataset
    
if __name__=='__main__':
    for data in ilc_dataset():
        print(data)
        break

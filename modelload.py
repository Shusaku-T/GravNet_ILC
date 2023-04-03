import torch
import torch
from torch_geometric.data import DataLoader
import numpy as np
import tqdm
from time import strftime
import os, os.path as osp
import uuid

#from datasets import tau_dataset, single_photon_dataset                        
from ReadText import ReadText
from datasets import ilc_dataset
from torch_cmspepr.gravnet_model import GravnetModel,GravnetModelWithNoiseFilter
#import torch_cmspepr.objectcondensation as oc                                  
import objectcondensation as oc

from matching import match, group_matching
from colorwheel import ColorWheel, HighlightColorwheel, ColorwheelWithProps

import event_view_plot as plt_3d
import warnings

def get_model():
    #model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)     
    #ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar'                        
    model=GravnetModel(input_dim=4,output_dim=3,k=50)
    ckpt = ReadText("Grav_ILC_setting.txt")["Output Model File"]
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
    return model

def main():
    model = get_model()
    print(model.state_dict()[list(model.state_dict().keys())[12]])
    
    

if __name__=="__main__":
    main()

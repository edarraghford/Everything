from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import torchvision.transforms.functional as TF
from PIL import Image 

import warnings
warnings.filterwarnings("ignore")

class SimulationsDataset(Dataset):

    def __init__(self, image_file, target_file): #, train=True):
        with open(image_file) as file:
            lines = [line.split()[0] for line in file]
        clust = []
        obs = [] 
        for i in range(len(lines)):
            clust.append(lines[i].split('/',1)[0])
            obs.append(lines[i].split('/',1)[1])
        self.clust = clust 
        self.obs = obs 
        self.target_file = target_file 

    def transform(self, image): 
        image = TF.center_crop(image, 180)
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.1307,), (0.3081,))

        return image 
        
    def __getitem__(self,index): 
        im_name = 'images/smoothed_img' + self.clust[index] + self.obs[index] + '.png'
        image = Image.open(im_name)
        image = image.convert('L')
        image = image.resize((360,360))
        image = self.transform(image)

        labels = np.loadtxt(self.target_file) 
        label = (labels[index]) 
        return image, label  
    
    def __len__(self): 
        return len(self.clust) 



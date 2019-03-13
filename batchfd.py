#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import imageio
import cv2
import os
from imgtools import rgb_2_darter
from fracdim import frac_dim

def get_files(path):
    folders = []
    index = {}
    os.chdir(path)
    
    for item in os.listdir():
        if os.path.isdir(item): folders.append(item)
    
    for folder in folders:
        os.chdir(path + '/' + folder)
        fish = []
        for item in os.listdir():
            if (item.endswith('.tiff') or item.endswith('.tif')):
                fish.append(item)
        index[folder] = fish

    return folders, index

def get_fd(path, selection = None, resize_dim = None, sample_dim = None):
    folders, index = get_files(path)

    def load_image(file):
        image = None

        if (selection != None and file[10] == selection):
            try:
                image = imageio.imread(file)
            except:
                print("Error: could not open file: ", file)
        elif (selection == None):
            try:
                image = imageio.imread(file)
            except:
                print("Error: could not open file: ", file)
        return image 

    #Determine the largest possible sample size for a square sample
    if (sample_dim == None):
        sample_dim = 0
        for folder in folders:
            os.chdir(path + '/' + folder)
            dims = []

            for file in index[folder]:
                image = load_image(file)
                if (image is not None): dims.append(np.min(image.shape[0:1]))
                else: continue
            if (np.min(dims) <= sample_dim or sample_dim == 0): sample_dim = np.min(dims)

    data = pd.DataFrame()

    for folder in folders:
        os.chdir(path + '/' + folder)
        D_vals = []

        #Go through list of files and calculate power spectrum for each image
        for file in index[folder]:
            print('Processing file: ', file, end = '\r')
            image = load_image(file)
            if (image is None): continue
            if (resize_dim is not None): image = cv2.resize(image, (resize_dim[0], resize_dim[1]))

            #Define a random square of size sample_domain x sample_domain from image
            sample_domain = np.subtract(image.shape[0:2], (sample_dim, sample_dim))       
            if sample_domain[0] == 0: sample_x = 0
            else: sample_x = np.random.randint(0, sample_domain[0])
            if sample_domain[1] == 0: sample_y = 0
            else: sample_y = np.random.randint(0, sample_domain[1])

            #Sample image and convert to darter color model
            image = image[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]
            image_modl = rgb_2_darter(image)
            image = image_modl[:, :, 0] + image_modl[:, :, 1]
            image = image / np.max(image)
            image = image * 256

            D = frac_dim(image)
            D_vals.append(D)
        
        data[folder] = pd.Series(D_vals)
    
    return data



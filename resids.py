#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code used to generate figures
'''

import numpy as np
import pandas as pd
import seaborn as sns
import imageio
import os
import cv2
import csv

from scipy import misc, optimize
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from numpy.fft import fft2, fftshift

from pspec import *

# Resizing parameters
fish_sample = 250

# Sampling parameters
fish_range = (10, 125)

base_dir = '~/Projects/Fourier-Analysis/Stats/'

#Darter image processing
path = '/home/mlab/Projects/Fourier-Analysis/Images/Crops'
folders = os.listdir(path)
counter = 0

skipped = 0
shape=[]
resids = np.zeros([19,576])
vals = [15,19]
for folder in folders:
    current_path = os.path.join(path, folder)
    files = os.listdir(current_path)

    for image in files:
        if image.endswith('.tif'):

            img_path = os.path.join(current_path, image)
            img = imageio.imread(img_path)
            shape.append(img.shape)

            sample_dim=fish_sample

            if (np.min(img.shape[0:2]) < fish_sample):
                skipped = skipped+1
                continue

            sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
            if sample_domain[0] == 0: sample_x = 0
            else: sample_x = np.random.randint(0, sample_domain[0])
            if sample_domain[1] == 0: sample_y = 0
            else: sample_y = np.random.randint(0, sample_domain[1])
            sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

            x, y = get_pspec(sample, bin_range=fish_range, return_bins=True)

            model = np.poly1d(np.polyfit(x[vals[0]:vals[1]], y[vals[0]:vals[1]], 1))

            for i in range(len(x)):
                resids[i, counter] = model(x[i]) - y[i]

            counter = counter + 1

from scipy import stats
F = stats.f_oneway(
resids[16,:],
resids[17,:],
resids[18,:])

means = np.mean(resids, axis=1)
error = np.std(resids, axis=1)
x = np.linspace(0, 1, 19)

plt.errorbar(x, means, yerr=error)


# Resizing parameters
hab_size = (600, 900)
hab_sample = 400
fish_size = (200, 200)
fish_sample = 200

# Sampling parameters
hab_range = (10, 220)
fish_range = (10, 110)
n_samples = 1


#Habitat image processing
path = './Images/Catagorized'
folders = os.listdir(path)

slopes = []
habitat = []

n_files = 0
for folder in folders:
    current_path = os.path.join(path, folder)
    files = os.listdir(current_path)
    n_files = n_files + len(files)  

counter = 0


current_file = 0
resids = np.zeros([19,n_files + 1])

for folder in folders:
    current_path = os.path.join(path, folder)
    files = os.listdir(current_path)

    for image in files:
        if image.endswith('.tiff'):
            img_path = os.path.join(current_path, image)
            img = imageio.imread(img_path)
            img = cv2.resize(img, dsize=hab_size, interpolation=cv2.INTER_LINEAR)

            status = 100*(current_file/n_files)
            print('Processing habitat images: ' + "%.2f" % round(status,2) + '% complete', end='\r')

            for i in range(n_samples):
                sample_dim=hab_sample
                sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
                if sample_domain[0] == 0: sample_x = 0
                else: sample_x = np.random.randint(0, sample_domain[0])
                if sample_domain[1] == 0: sample_y = 0
                else: sample_y = np.random.randint(0, sample_domain[1])
                sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

                x, y = get_pspec(sample, bin_range=fish_range, return_bins=True)

                model = np.poly1d(np.polyfit(x, y, 1))

                for i in range(len(x)):
                    resids[i, counter] = model(x[i]) - y[i]

                counter = counter + 1

means = np.mean(resids, axis=1)
error = np.std(resids, axis=1)
x = np.linspace(0, 1, 19)

plt.errorbar(x, means, yerr=error)

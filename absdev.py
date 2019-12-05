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
hab_size = (600, 900)
hab_sample = 200
fish_size = (600, 600)
fish_sample = 200

# Sampling parameters
hab_range = (10, 210)
fish_range = (10, 210)
n_samples = 2

base_dir = '~/Projects/Fourier-Analysis/Stats/'

hab_data = pd.read_csv(base_dir + 'habitats600.csv')
hab_data = hab_data.groupby(['habitat']).mean().values


hab_means = {}
boulder = []
gravel = []
bedrock = []
sand = []
detritus = []

hab_means['boulder'] = hab_data[1]
hab_means['gravel'] = hab_data[3]
hab_means['bedrock'] = hab_data[0]
hab_means['sand'] = hab_data[4]
hab_means['detritus'] = hab_data[2]

devs = []
species = []
habitat = []
sexes = []
hab_slopes = []
animal = []
sites = []

habitats = {}
habitats['barrenense'] = 'bedrock'
habitats['blennioides'] = 'boulder'
habitats['caeruleum'] = 'gravel'
habitats['camurum'] = 'boulder'
habitats['chlorosomum'] = 'sand'
habitats['gracile'] = 'detritus'
habitats['olmstedi'] = 'sand'
habitats['pyrrhogaster'] = 'sand'
habitats['swaini'] = 'detritus'
habitats['zonale'] = 'gravel'

animals = {}
animals['barrenense'] = 'Etheostoma_barrenense_A'
animals['blennioides'] = 'Etheostoma_blennioides_A'
animals['caeruleum'] = 'Etheostoma_caeruleum_A'
animals['camurum'] = 'Nothonotus_camurus_A'
animals['chlorosomum'] = 'Etheostoma_chlorosoma_A'
animals['gracile'] = 'Etheostoma_gracile_A'
animals['olmstedi'] = 'Etheostoma_olmstedi_C'
animals['pyrrhogaster'] = 'Etheostoma_pyrrhogaster_A'
animals['swaini'] = 'Etheostoma_swaini_A'
animals['zonale'] = 'Etheostoma_zonale_A'

def sse_dev(bin_x, bin_y, slope):
    bin_x = np.log(bin_x)
    bin_y = np.log(bin_y)
    def lineq(x, m, b):
        return m*x + b

    def mse(b):
        return np.mean(np.power(bin_y - lineq(bin_x, slope, b), 2))

    b = optimize.minimize_scalar(mse).fun

    return b

#Darter image processing
path = '/home/mlab/Projects/Fourier-Analysis/Images/Crops'
folders = os.listdir(path)

for folder in folders:
    current_path = os.path.join(path, folder)
    files = os.listdir(current_path)

    for image in files:
        if image.endswith('.tif'):
            img_path = os.path.join(current_path, image)
            img = imageio.imread(img_path)

            for i in range(n_samples):
                sample_dim=fish_sample
                sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
                if sample_domain[0] == 0: sample_x = 0
                else: sample_x = np.random.randint(0, sample_domain[0])
                if sample_domain[1] == 0: sample_y = 0
                else: sample_y = np.random.randint(0, sample_domain[1])
                sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

                sample = cv2.resize(sample, dsize=fish_size, interpolation=cv2.INTER_LINEAR)

                hab_slope = hab_means[habitats[folder]]

                x, y = get_pspec(sample, bin_range=fish_range, return_bins=True)
                slope_dev = sse_dev(x, y, hab_slope)
                print(slope_dev)

                devs.append(slope_dev)
                species.append(folder)
                habitat.append(habitats[folder])
                sexes.append(image[-7])
                hab_slopes.append(hab_means[habitats[folder]])
                animal.append(animals[folder])
                sites.append(image[-12:-8])
    
with open('./fishdevs600.csv', mode='w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['animal', 'habitat', 'sex', 'devs', 'hab_slope', 'site'])
    for i in range(len(devs)):
        writer.writerow([species[i],
            animal[i], 
            habitat[i],
            sexes[i],
            devs[i],
            hab_slopes[i],
            sites[i]])
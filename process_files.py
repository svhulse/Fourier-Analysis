#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code used load all images, calculate their Fourier power specturm, and save the
data in a format which is easy to run statistics with. Folders locations and
file structure are hard coded, and must be changed for different file structures
'''

import numpy as np
import imageio
import os
import csv
import cv2

from pspec import get_pspec

# Resizing parameters
hab_size = (1024, 1024)
hab_sample = 512
fish_size = (512, 512)
fish_sample = 200

# Sampling parameters
hab_range = (10, 255)
fish_range = (10, 255)
n_samples = 4

__author__ = 'Samuel Hulse'
__email__ = 'hsamuel1@umbc.edu'

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

def random_sample(data, sample_dim):
    sample_domain = np.subtract(data.shape[0:2], (sample_dim, sample_dim))       
    if sample_domain[0] == 0: sample_x = 0
    else: sample_x = np.random.randint(0, sample_domain[0])
    if sample_domain[1] == 0: sample_y = 0
    else: sample_y = np.random.randint(0, sample_domain[1])

    sample = data[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

    return sample

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

current_file = 0
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

                slope = get_pspec(sample, bin_range=hab_range)
                slopes.append(slope)
                habitat.append(folder)

            current_file = current_file + 1

hab_means = {}
boulder = []
gravel = []
bedrock = []
sand = []
detritus = []

for i in range(len(slopes)):
    if habitat[i] == 'boulder':
        boulder.append(slopes[i])
    elif habitat[i] == 'gravel':
        gravel.append(slopes[i])
    elif habitat[i] == 'bedrock':
        bedrock.append(slopes[i])
    elif  habitat[i] == 'sand':
        sand.append(slopes[i])
    elif habitat[i] == 'detritus':
        detritus.append(slopes[i])

hab_means['boulder'] = np.mean(boulder)
hab_means['gravel'] = np.mean(gravel)
hab_means['bedrock'] = np.mean(bedrock)
hab_means['sand'] = np.mean(sand)
hab_means['detritus'] = np.mean(detritus)

with open('../habitats.csv', mode='w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['habitat', 'slope'])
    for i in range(len(slopes)):
        writer.writerow([habitat[i], slopes[i]])

slopes = []
species = []
habitat = []
sexes = []
hab_slopes = []
animal = []
sites = []

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

            sample_dim=fish_sample
            sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
            if sample_domain[0] == 0: sample_x = 0
            else: sample_x = np.random.randint(0, sample_domain[0])
            if sample_domain[1] == 0: sample_y = 0
            else: sample_y = np.random.randint(0, sample_domain[1])
            sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

            sample = cv2.resize(sample, dsize=fish_size, interpolation=cv2.INTER_LINEAR)

            slope = get_pspec(sample, bin_range=fish_range)

            slopes.append(slope)
            species.append(folder)
            habitat.append(habitats[folder])
            sexes.append(image[-7])
            hab_slopes.append(hab_means[habitats[folder]])
            animal.append(animals[folder])
            sites.append(image[-12:-8])
    
with open('../fish.csv', mode='w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['species', 'animal', 'habitat', 'sex', 'slope', 'hab_slope', 'site'])
    for i in range(len(slopes)):
        writer.writerow([species[i],
            animal[i], 
            habitat[i],
            sexes[i],
            slopes[i],
            hab_slopes[i],
            sites[i]])

            

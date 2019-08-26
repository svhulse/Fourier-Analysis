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

from scipy import misc
from pspec import get_pspec

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
path = 'C:/Users/renoult/Desktop/Sam/Catagorized'
folders = os.listdir(path)

slopes = []
habitat = []
n_samples = 2

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
            img = misc.imresize(img, (800, 1200))

            status = 100*(current_file/n_files)
            print('Processing habitat images: ' + "%.2f" % round(status,2) + '% complete', end='\r')
            for i in range(n_samples):
                sample_dim=400
                sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
                if sample_domain[0] == 0: sample_x = 0
                else: sample_x = np.random.randint(0, sample_domain[0])
                if sample_domain[1] == 0: sample_y = 0
                else: sample_y = np.random.randint(0, sample_domain[1])
                sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

                slope = get_pspec(sample, bin_range=(10, 200))
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
path = 'C:/Users/renoult/Desktop/Sam/Crops'
folders = os.listdir(path)

for folder in folders:
    current_path = os.path.join(path, folder)
    files = os.listdir(current_path)

    for image in files:
        if image.endswith('.tif'):
            img_path = os.path.join(current_path, image)
            img = imageio.imread(img_path)

            sample_dim=200
            sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
            if sample_domain[0] == 0: sample_x = 0
            else: sample_x = np.random.randint(0, sample_domain[0])
            if sample_domain[1] == 0: sample_y = 0
            else: sample_y = np.random.randint(0, sample_domain[1])
            sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

            slope = get_pspec(sample, bin_range=(10,110))

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

            
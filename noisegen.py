#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Method for generating white noise filtered to have a specified Fourier power
spectrum slope
'''

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

__author__ = 'Sam Hulse'
__email__ = 'hsamuel1@umbc.edu'

def gen_noise(slope, dim, whiten = False):
    #Create matrix of standard normal values
    n_samples = dim[0]*dim[1]
    noise = np.random.normal(0, 1, n_samples)
    noise = np.reshape(noise, dim)

    noise_fft = fftshift(fft2(noise))
    fft_dim = noise_fft.shape
    center = np.divide(fft_dim, 2)

    #Create distance bins for applying the filter, modify the multiplier on the second line to change granularity 
    max_dist = np.sqrt(np.sum(np.square(center)))
    n_bins = int(np.ceil(max_dist)) * 2
    bins = np.linspace(0, max_dist, n_bins)

    #Generate euclidean distance matrix 
    dist_matrix = np.zeros(fft_dim)
    x_sample = np.linspace(0, fft_dim[0], fft_dim[0])
    y_sample = np.linspace(0, fft_dim[1], fft_dim[1])
    x, y = np.meshgrid(y_sample, x_sample)
    
    x = np.absolute(x - center[1])
    y = np.absolute(y - center[0])
    dist_matrix = np.sqrt(x**2 + y**2)

    #Apply filter to each radial slice
    filtered_noise = np.zeros(fft_dim)
    for i in range(len((bins[0:-1]))):            
        filter = np.ones(fft_dim)
        filter[dist_matrix <= bins[i]] = 0
        filter[dist_matrix > bins[i+1]] = 0

        if whiten:
            bla = img_fft * filter  
            hur = img_fft * (1 - filter)
            

        power = np.mean([bins[i], bins[i+1]])**(slope/2)
        filtered_noise = filtered_noise + noise_fft*(filter * power)
    
    filtered_noise = np.absolute(ifft2(ifftshift(filtered_noise)))
    return filtered_noise

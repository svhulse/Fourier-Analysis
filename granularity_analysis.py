#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import imageio
import os
from numpy import fft2, ifft2, fftshift, ifftshift

__author__ = 'Sam Hulse'
__email__ = 'hsamuel1@umbc.edu'

def get_stdev(image, scales):
    img = imageio.imread(image)
    img_lum = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    img_lum = img_lum / np.max(img_lum)

    img_fft = fftshift(fft2(img_lum))
    pattern_energy = []

    dist_matrix = np.zeros(img_fft.shape)
    x_sample = np.linspace(0, fft_dim[0], fft_dim[0])
    y_sample = np.linspace(0, fft_dim[1], fft_dim[1])
    x, y = np.meshgrid(y_sample, x_sample)
    
    x = np.absolute(x - center[1])
    y = np.absolute(y - center[0])
    dist_matrix = np.sqrt(x**2 + y**2)

    for filt_num in range(len(scales) - 1):
        center = np.divide(img_fft.shape, 2)
        filter = np.ones(img_fft.shape)
        filter[dist_matrix <= scales[filt_num]] = 0
        filter[dist_matrix > scales[filt_num + 1]] = 0

        img_fft_filt = img_fft * filter
        img_filt = ifft2(ifftshift(img_fft_filt))
        pattern_energy.append(np.std(img_filt))
        filename = ('filters/' + image[0:-4] + '_' + str(scales[filt_num]) + '.tiff')
    
    return pattern_energy

def batch_process(path, scales):
    os.chdir(path)
    if not(os.path.isdir('filters')):
        os.mkdir('filters')

    for item in os.listdir():
        if item.endswith('.tif'):
            energy = get_stdev(item, scales)
            print('File: ' + item)
            max_energy = np.argmax(energy)
            print('Maximum energy is '
                + str(energy[max_energy]) 
                + ' at scale ' 
                + str(scales[max_energy]) + ' to '
                + str(scales[max_energy + 1]) + 'pixels')

if (__name__ == '__main__'):
    scales = [0, 4, 16, 32, 64]
    batch_process('/home/sam/Desktop/Images')
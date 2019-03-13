#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Methods for calculating the slope of the 2-Dimensional Fourier transform of an image
'''

import numpy as np
from numpy.fft import fft2, fftshift, rfft2, ifftshift

__author__ = 'Sam Hulse'
__email__ = 'hsamuel1@umbc.edu'

def rotavg(data):
    center = np.divide(data.shape, 2)
    x_sample = np.linspace(0, data.shape[0], data.shape[0])
    y_sample = np.linspace(0, data.shape[1], data.shape[1])
    x, y = np.meshgrid(y_sample, x_sample)
    
    x = np.absolute(x - center[1])
    y = np.absolute(y - center[0])
    dist_matrix = np.sqrt(x**2 + y**2)

    max_dist = np.sqrt(np.sum(np.square(center)))
    n_bins = int(np.ceil(max_dist))
    bins = np.linspace(0, max_dist, n_bins)
    
    radialprofile = np.zeros(n_bins - 1)
    
    for i in range(len((bins[0:-1]))):
        filter = np.ones(data.shape)
        filter[dist_matrix <= bins[i]] = 0
        filter[dist_matrix > bins[i+1]] = 0
        
        radialprofile[i] = np.sum(filter * data) / np.sum(filter)
    
    return radialprofile
        

'''
def rotavg(data):
    center = np.array(data.shape) * 0.5
    y, x = np.indices(data.shape)

    rho = np.sqrt((x - center[0]) ** 2 + (y - center[1])**2)
    rho = rho.astype(np.int)
    tbin = np.bincount(rho.ravel(), data.ravel())
    nr = np.bincount(rho.ravel())

    radialprofile = tbin / nr
    return radialprofile
'''

def imfft(image):
    imfft = fftshift(rfft2(image))
    impfft = np.absolute(imfft) ** 2
    pspec = rotavg(impfft)

    return pspec

def comp_slope(pspec):
    pspec = np.log(pspec)
    x = np.linspace(1, pspec.shape[0] / 2, pspec.shape[0])
    x = np.log(x)
    slope = np.polyfit(x, pspec, 1)[0]
    return slope

#!/usr/bin/env python

''' pspec.py: Contains methods used to calculate the slope of the Fourier power 
spectrum of image files.

This also allows for the conversion of RGB images to
our dichromatic darter color vision model. This conversion is assuming that the
images were captured with a Canon EOS 5D mark iv, and extracted from RAW files
using no white balance, or gamma correction.

'''

__author__ = 'Samuel Hulse'
__email__ = 'hsamuel1@umbc.edu'

import numpy as np

from numpy.fft import fft2, fftshift

def get_pspec(image, 
	bin_range=(10, 110), 
	kaiser=True, 
	color_model=True, 
	n_bins=20):
	''' Primary image handling method

	Args:
		image (int NxNx3): input image.
		bin_range (int 2-tuple): frequency range in which to calculate the
			slope.
		kaiser (bool): whether to apply Kaiser-Borel window to input image.
		color_model (bool): whether to apply darter color model, if false
			convert image to grayscale.
		n_bins (int): number of binln -s to be sampled within bin_range for slope
			computation.
	
	Returns:
		slope (int): The slope of the Fourier power spectrum.

	'''

	#Apply darter color model if enabled, otherwise apply standard RGB to
	#luminance transformation
	if color_model:
		image = rgb_2_darter(image)
		image = image[:, :, 0] + image[:, :, 1]
		image = image / np.max(image)
	else:
		image = 0.21*image[:, :, 0] + 0.72*image[:, :, 1] + 0.07*image[:, :, 2]

	#Apply the Kaiser-Borel transformation if enabled
	if kaiser:
		image = kaiser2D(image, 2)

	#Calculate the power spectrum and its slope
	pspec = imfft(image)
	x, y = bin_pspec(pspec, n_bins, bin_range)
	slope = get_slope(x, y)

	return slope
	
def rgb_2_darter(image):
	''' Covert RGB image to dichromatic darter color model

	Args:
		image (int NxNx3): input image
	
	Returns:
		im_out (int NxNx2): image converted to darter color model
	'''

	im_out = np.zeros([image.shape[0], image.shape[1], 3], dtype = np.float32)

	im_out[:, :, 1] = (140.7718694130528 +
		0.021721843447502408  * image[:, :, 0] +
		0.6777093385296341    * image[:, :, 1] +
		0.2718422677618606    * image[:, :, 2] +
		1.831294521246718E-8  * image[:, :, 0] * image[:, :, 1] +
		3.356941424659517E-7  * image[:, :, 0] * image[:, :, 2] +
		-1.181401963067949E-8 * image[:, :, 1] * image[:, :, 2])
	im_out[:, :, 0] = (329.4869869234302 +
		0.5254935133632187    * image[:, :, 0] +
		0.3540642397052902    * image[:, :, 1] +
		0.0907634883372674    * image[:, :, 2] +
		9.245344681241058E-7  * image[:, :, 0] * image[:, :, 1] +
		-6.975682782165032E-7 * image[:, :, 0] * image[:, :, 2] +
		5.828585657562557E-8  * image[:, :, 1] * image[:, :, 2])

	return im_out

def rotavg(data):
	''' Compute rotational average for a matrix

	Args:
		data (float NxN): input matrix.
	
	Returns:
		rot_avg (float vector): rotational average with rot_avg[0] being the
			center. Length is equal to the maximum distance of all points to the
			center.

	'''

	#Generate a euclidean distance matrix with the same shape as the input
	center = np.divide(data.shape, 2)
	x_sample = np.linspace(0, data.shape[0], data.shape[0])
	y_sample = np.linspace(0, data.shape[1], data.shape[1])
	x, y = np.meshgrid(y_sample, x_sample)
	
	x = np.absolute(x - center[1])
	y = np.absolute(y - center[0])
	dist_matrix = np.sqrt(x**2 + y**2)

	#Find the maximum distance from the center for all points
	max_dist = np.sqrt(np.sum(np.square(center)))
	n_bins = int(np.ceil(max_dist))
	bins = np.linspace(0, max_dist, n_bins)
	
	rot_avg = np.zeros(n_bins - 1)
	
	#For each integer distance find all values which fall within that range and
	#calculate their average
	for i in range(len(bins[0:-1])):
		filter = np.zeros(data.shape)
		filter[np.logical_and(
			dist_matrix >= bins[i], 
			dist_matrix < bins[i+1])] = 1
		
		rot_avg[i] = np.sum(filter * data) / np.sum(filter)
	
	return rot_avg

def bin_pspec(data, n_bins, bin_range):
	''' Sample a power spectrum across a fixed frequency range at at intervals
	which will be evenly distributed when log transformed

	Args:
		data (float vector): input power spectrum.
		n_bins (int): number of bins to sample.
		bin_range (int 2-tuple): start frequency and end frequency to sample the
			power spectrum.

	Returns:
		bin_x (float vector): frequencies where bins were sampled
		bin_y (float vector): power spectrum values for each bin

	'''

	#Generate logarithmic coordinate axes to sample bins
	bins = np.logspace(np.log(bin_range[0]), 
		np.log(bin_range[1]),
		n_bins,
		base = np.e)
	x = np.linspace(1, len(data), len(data))

	bins_x = []
	bins_y = []

	#Calculate the values and coordinates for each bin by taking the average of
	#all log-transformed data points which fall between each bin
	for i in range(len(bins[0:-1])):
		bin_x = np.mean(x[np.logical_and(
			x >= bins[i],
			x < bins[i+1])])
		bins_x.append(bin_x)

		bin_y = np.mean(data[np.logical_and(
			x >= bins[i],
			x < bins[i+1])])
		bins_y.append(bin_y)
	
	return (bins_x, bins_y)

def get_slope(x, y):
	''' Compute the slope of a power spectrum, simply log-transforms input data
	and returns the slope of a linear regression of the log-transformed input

	Args:
		x (float vector): x-values for slope computation.
		y (float vector): y-values for slope computation.

	Returns:
		slope (float): The slope of the linear regression.
	
	'''

	x, y = np.log((x, y))
	slope = np.polyfit(x, y, 1)[0]

	return slope

def kaiser2D(img, alpha):
	''' Apply a Kaiser-Borel window to an input image. This method has been
	shown to reduce edge effects for Fourier analysis of images

	Args:
		img (float NxN): input grayscale image.
		alpha (int): strength parameter for transform.
	
	Returns:
		im_out (float NxN): transformed image.

	'''

	#Calculate the 2D Kaiser-Bessel window as the outer product of the 1D
	#window with itself
	kaiser = np.kaiser(img.shape[0], alpha*np.pi)
	A = np.outer(kaiser, kaiser)
	
	#Normalize by the sum of squared weights
	w = np.sum(A*A)
	A = A / w

	#Apply the window by performing elementwise multiplication
	im_out = np.multiply(img, A)
	return im_out

def imfft(image):
	''' Calculate the rotationally-averaged Fourier power spectrum of an input
	image

	Args:
		image (float NxN): grayscale input image.

	Returns:
		pspec (float vector): rotationally-averaged Fourier power spectrum.

	'''

	imfft = fftshift(fft2(image))
	impfft = np.absolute(imfft) ** 2
	pspec = rotavg(impfft)

	return pspec

import numpy as np

from numpy.fft import fft2, fftshift

def get_pspec(image, bin_range=(10, 110), kaiser=True, color_model=True, n_bins=20):
    
    if color_model:
        image = rgb_2_darter(image)
        image = image[:, :, 0] + image[:, :, 1]
        image = image / np.max(image)

    if kaiser:
        image = kaiser2D(image, 2)

    pspec = imfft(image)
    x, y = bin_pspec(pspec, n_bins, bin_range)
    slope = get_slope(x, y)

    return slope
    
def rgb_2_darter(image):
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
    
    for i in range(len(bins[0:-1])):
        filter = np.zeros(data.shape)
        filter[np.logical_and(
            dist_matrix >= bins[i], 
            dist_matrix < bins[i+1])] = 1
        
        radialprofile[i] = np.sum(filter * data) / np.sum(filter)
    
    return radialprofile

def bin_pspec(data, n_bins, bin_range):
    bins = np.logspace(np.log(bin_range[0]), 
        np.log(bin_range[1]),
        n_bins,
        base = np.e)
    x = np.linspace(1, len(data), len(data))

    bins_x = []
    bins_y = []

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
    x, y = np.log((x, y))
    slope = np.polyfit(x, y, 1)[0]

    return slope

def kaiser2D(img, alpha):
    #Calculate the 2D Kaiser-Bessel window as the outer product of the 1D window with itself
    kaiser = np.kaiser(img.shape[0], alpha*np.pi)
    A = np.outer(kaiser, kaiser)
    
    #Normalize by the sum of squared weights
    w = np.sum(A*A)
    A = A / w

    #Apply the window by performing elementwise multiplication
    imout = np.multiply(img, A)
    return imout

def imfft(image):
    imfft = fftshift(fft2(image))
    impfft = np.absolute(imfft) ** 2
    pspec = rotavg(impfft)

    return pspec

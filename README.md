# Data Files
All data used for Hulse et al., appears in this repository. For each file, the number after the underscore represents the image dimensions used to generate the data set. For all analyses presented in the paper, we used the 200x200 scale images.

# Image Processing
The following code was written to process Canon CR2 Raw files while maintaning high dynamic range, and linearity. The file process_linear_raw.py takes CR2 files as inputs and saves them as .tif files. Since they are uncompressed, they can be quite large. Additionally, the output images have no white balance applied, and due to the higher concentration of green sensors, appear much greener than normal images.

# Fourier Analysis
Calculate the slope of the Fourier power spectrum for images. All methods required to calculate the slope of the fourier power spectrum are contained in pspec.py. Images must be square, and RGB triplets, formatted as numpy arrays. To calculate the power spectrum, call the get_pspec function

`from pspec import get_pspec`

`image = imageio.imread("example.jpg")`

`image = image[0:400, 0:400, :]`

`slope = get_pspec(image)`

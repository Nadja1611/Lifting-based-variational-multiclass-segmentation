
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import skimage.io
import os
import scipy
from skimage.transform import resize


os.chdir('D:\Chan Vese Algorithm\Code_sauber')
           
im = plt.imread('textur_3channel.png')
#im = plt.imread("5-channel_texture.png")
grayscale = rgb2gray(im[:,:,:3])
img = grayscale
img1 = img
#img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
img1=resize(img1,(256,256))
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


#shrink = (slice(0, None, 1), slice(0, None, 1))
brick = img_as_float(img1)#

images = brick

# prepare reference features
ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(brick, kernels)




def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernels=[]
kernel_params = []
for theta in (0, 1,2,3):
    theta = theta / 4. * np.pi
    for k in range(2,9):
        frequency =np.sqrt(2)* 2**k/256
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append(power(img1, kernel))
        kernels.append(kernel)


#values for 3channel brodatz for paper
A = results[3]+results[4]+results[7]+results[21]
B=results[9]+results[8]+results[24]+results[22]
C= results[11]+results[12]+results[18]+results[26]

from scipy.ndimage import gaussian_filter
A=gaussian_filter(A, sigma=16)
B=gaussian_filter(B, sigma=6)
C=gaussian_filter(C, sigma=6)


# #values for 5 channel
# A=results[16]+results[2]+results[9]+results[23]
# B= results[4]+results[10]+results[11]+results[24]
# C=results[3]+results[11]
# D=1-A-B-C
# E=np.copy(img)
# E[E<0.98]=0



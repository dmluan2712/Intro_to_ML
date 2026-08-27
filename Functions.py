import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

# Define functions for displaying images
def plot_image(img: np.array):
    plt.figure(figsize = (6, 6))
    plt.imshow(img, cmap = 'gray')

def plot_two_images(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize = (12 ,6))
    ax[0].imshow(img1, cmap = 'gray')
    ax[1].imshow(img2, cmap = 'gray');
    
# Calculate number of windows
def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0
    
    for i in range(img_size):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1
    return num_pixels
    
# You can implement convolution over the entire 
# image using the convolve function
def convolve(img: np.array, kernel: np.array) -> np.array:
    tgt_size = calculate_target_size(
        img_size = img.shape[0],
        kernel_size = kernel.shape[0]
    )
    k = kernel.shape[0]
    convolved_img = np.zeros(shape = (tgt_size, tgt_size))
    
    for i in range(tgt_size):
        for j in range(tgt_size):
            mat = img[i:i+k, j:j+k]
            convolved_img[i,j] = np.sum(np.multiply(mat, kernel))
    return convolved_img
import numpy as np
import cv2
import os
import scipy
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def generate_data_franke(N=500, defect=True):    
    x = np.arange(N)/N
    m1 = np.sin(np.pi*2*x)
    m2 = m1.copy()
    if defect:
        m2 += np.exp(-800*np.square(x-0.5))
    sigma = 0.7 - 1.4*np.square(x-0.5)
    y1, y2 = generate_synthetic_data(m1, m2, sigma)
    return y1, y2

def bartlett_priestley_kernel(u, h):
    """
    implements rescaled bartlett priestley kernel
    """
    return np.maximum((1/h) * (3/4) * (1- np.power(u / h, 2)), 0)

def gaussian_kernel(u, h):
    """
    implements rescaled gaussian kernel
    """
    return (1/h) * (1 / np.sqrt(2*np.pi)) * np.exp((-1/2)*np.power(u / h, 2))

def generate_synthetic_data(m1, m2, sigma):
    """ 
    All inputs should be 1d np arrays of the same shape 
    # TODO Try other distributions
    """
    y1 = m1 + np.random.standard_normal(m1.shape[0]) * sigma
    y2 = m2 + np.random.standard_normal(m1.shape[0]) * sigma
    
    return y1, y2

def calc_smoothed_estimate(y, kernel_function, h):
    """ 
        This function implements a smoothed average estimator, using the kernel_function for smoothing.

        Arguments :
            y : 1D np-array of the shape NxP
            kernel_function : Real-valued kernel function 

        Returns :
            m : Smoothed estimate of y        
    """
    N = y.shape[0]
    m = np.zeros_like(y)
    for i in range(N): # TODO Optimize
        x_i = np.arange(N)/N 
        kernel = kernel_function(i/N - x_i, h)
        m[i] = np.mean(kernel*y)
    return m

def calc_smoothed_estimate_parallel(y, kernel_function, h):
    """ 
        This function implements a smoothed average estimator, using the kernel_function for smoothing.

        Arguments :
            y : 1D np-array of the shape NxP
            kernel_function : Real-valued kernel function 

        Returns :
            m : Smoothed estimate of y        
    """
    N = y.shape[1]
    m = np.zeros_like(y)
    for i in range(N): # TODO Optimize
        x_i = np.arange(N)/N
        kernel = kernel_function(i/N - x_i, h)
        m[:,i] = np.mean(np.tile(kernel,(y.shape[0],1)) *y, axis=1)
    return m

def calc_Tn(m1, m2, h, axis=0):
    """
    implements test statistic
    """
    return np.sqrt(h) * np.sum(np.square(m1-m2), axis=axis)


def load_images(folders, filenames, target_size=(100, 100), detrend=False, normalize=False):
    X, X_hat = [], []
    for folder in folders:
        files = filenames if filenames else os.listdir(folder)
        for filename in files:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, target_size)
                img = np.divide(img, 255)

                if detrend:
                # remove linear trends
                    img = scipy.signal.detrend(img, axis=0)
                    img = scipy.signal.detrend(img, axis=1)
                if normalize:
                    # max scaling
                    img /= np.max(img)

                if folder == "defect_images":
                    X_hat.append(img)
                elif folder == "no_defect_images":
                    X.append(img)
    return X, X_hat


def plot_defect_area(defect_image, minpoint, maxpoint, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(defect_image, cmap = 'gray', vmin=0, vmax=1)
    rect = patches.Rectangle((minpoint[1], minpoint[0]), maxpoint[1] - minpoint[1], maxpoint[0] - minpoint[0], linewidth=3, edgecolor='r', fill=False)
    ax.add_patch(rect)
    return ax

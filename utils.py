import numpy as np
import cv2
import os

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

def calc_Tn(m1, m2, h):
    """
    implements test statistic
    """
    return np.sqrt(h) * np.sum(np.square(m1-m2))


def load_images(folders, target_size=(100, 100)):
    X, X_hat = [], []
    for folder in folders:
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, target_size)
                img = np.divide(img, 255)
                if folder == "defect_images":
                    X_hat.append(img)
                elif folder == "no_defect_images":
                    X.append(img)
    return X, X_hat




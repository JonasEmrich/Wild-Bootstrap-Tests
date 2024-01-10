import numpy as np

def generate_data_franke(N = 500):    
    x = np.arange(N)/N
    m1 = np.sin(np.pi*2*x)
    sigma = 0.7 - 1.4*np.square(x-0.5)
    y1, y2 = generate_synthetic_data(m1, m1, sigma)
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
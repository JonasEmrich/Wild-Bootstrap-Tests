import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import functools

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

def smoothed_estimate(y, kernel_function, h):
    """ 
        This function implements a smoothed average estimator, using the kernel_function for smoothing.
        See https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4286562 for the formula 

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

def calculate_Tn(m1, m2, h):
    return np.sqrt(h) * np.sum(np.square(m1-m2))

def get_wild_residual(epsilon_hat):
    """ 
        PLEASE INPUT VECTOR
    """
    p = np.random.random_sample(epsilon_hat.shape[0])
    gamma = (5 + np.sqrt(5)) / 10
    mask = p < gamma
    a = ((1-np.sqrt(5)) / 2) * epsilon_hat
    b = ((1+np.sqrt(5)) / 2) * epsilon_hat
    b[mask] = a[mask]
    return b

def get_wild_residual_normal(epsilon_hat):
    """ 
        PLEASE INPUT VECTOR
    """
    V_i = np.random.standard_normal(epsilon_hat.shape[0])
    return ((1/np.sqrt(2))*V_i + (1/2)*(np.square(V_i)-1))*epsilon_hat

def generate_synthetic_data(m1, m2, sigma):
    """ 
        All inputs should be 1d np arrays of the same shape # TODO Try other distributions
    """
    y1 = m1 + np.random.standard_normal(m1.shape[0]) * sigma
    y2 = m2 + np.random.standard_normal(m1.shape[0]) * sigma
    
    return y1, y2

def bootstrap_iteration(b, residual_function, epsilon_hat_1, epsilon_hat_2, m1_g, kernel_function, h):
    """
    Computes one Bootstrap iteration
    1. computing wild residuals
    2. computing resulting bootstrap observations y*
    3. estimating bootstrap smoothed estimates m*
    4. calculate bootstrap test statstistic Tn*
    """
    np.random.seed(b) # setting seed so that every iteration uses different random number generator starting points
    wild_epsilon_1 = residual_function(epsilon_hat_1)
    wild_epsilon_2 = residual_function(epsilon_hat_2)

    y1_star = m1_g + wild_epsilon_1
    y2_star = m1_g + wild_epsilon_2

    m1_star = smoothed_estimate(y1_star, kernel_function, h)
    m2_star = smoothed_estimate(y2_star, kernel_function, h)

    return calculate_Tn(m1_star, m2_star, h) # Tn_star


def wild_bootstrap(y1, y2, kernel_function, h, g, residual_function, B, alpha):
    """ 
        returns true if H0 is rejected

    """

    m1 = smoothed_estimate(y1, kernel_function, h)
    m2 = smoothed_estimate(y2, kernel_function, h)
    m1_g = smoothed_estimate(y1, kernel_function, g)

    Tn = calculate_Tn(m1, m2, h)

    epsilon_hat_1 = y1 - m1
    epsilon_hat_2 = y2 - m2

    # Parallel processing of one boostrap iteration
    Tn_star = []
    pool = mp.Pool(processes=mp.cpu_count()-1)
    func = functools.partial(bootstrap_iteration, 
                             residual_function=residual_function, 
                             epsilon_hat_1=epsilon_hat_1, 
                             epsilon_hat_2=epsilon_hat_2, 
                             m1_g=m1_g, 
                             kernel_function=kernel_function, 
                             h=h) # create auxiliary function
    Tn_star = np.array([result for result in tqdm(pool.imap(func, np.arange(B)), total=B)]) # collect results of B iterations
    pool.close()
    pool.terminate()
    pool.join()    

    q = 1-alpha
    c_alpha_star = np.quantile(Tn_star, q)

    rejected_bool = Tn > c_alpha_star

    print("The Hypothesis H0 was %srejected" %("" if rejected_bool else "not "))
    print("c_alpha_star is %.4f"%c_alpha_star)

    return Tn > c_alpha_star, c_alpha_star, Tn_star
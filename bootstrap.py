import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import stats

from utils import *


class Bootstrap():
    def __init__(self, kernel_function="bartlett_priestley", method="wild"):

        # set the kernel function
        if kernel_function in ["bartlett_priestley", "bartlett_priestley_kernel", "bartlett"]:
            self.kernel_function = bartlett_priestley_kernel
        elif kernel_function in ["gaussian", "gaussian_kernel"]:
            self.kernel_function = gaussian_kernel
        else:
            raise ValueError("Unknown kernel function.")

        # select residual function according to method
        if method in ["bootstrap", "classic", "normal", "resample_residual"]:
            self.residual_function = self._resample_residual
        elif method in ["wild", "wild_binary", "wild_pivotal", "wild_extended", "wild_extended_smoothed"]:
            self.residual_function = self._sample_wild_residual
        elif method in ["wild_normal"]:
            self.residual_function = self._sample_wild_residual_normal
        else:
            raise ValueError("Unknown method.")

    def compute(self, y1, y2, h=.02, g=.03, B=1000, B_std=25, alpha=[.05], beta=0.95, printout=True, show_progress = True):
        """
        performs the computation of the (wild) bootstrap test

        args:
            y1 (array):     the first (reference) observational signal
            y2 (array):     the second observational signal to test
            h (float):      the kernel bandwidth of the initial estimates
            g (float):      the kernel bandwidth of bootstrap estimates
            B (int):        number of Bootstrap iterations
            alpha (float):  the significance level

        returns:
            (bool):     True if the null hypothesis was rejected (defect) otherwise False (no defect)
            (float):    c_alpha_star - (1-alpha) quantile of bootstrap tests statistics
            (array):    Tn_star - bootstrap tests statistics
            (float):    Tn - initial estimate of test statistic
        """

        self.g = g
        self.h = h
        self.B = B
        self.B_std = B_std
        self.printout = printout
        self.show_progress = show_progress

        # compute initial estimates
        (m1, m2, m1_g), Tn = self._calc_init_estimates(y1, y2, self.h, self.g)

        # calculate residuals
        epsilon_hat_1 = y1 - m1
        epsilon_hat_2 = y2 - m2

        std = np.zeros_like(epsilon_hat_1)
        N = epsilon_hat_1.shape[0]
        for i in range(N):
            data = epsilon_hat_1[max(0,i-N//10):i+N//10]
            if(i > 0):
                std[i] = 0.95*std[i-1]+0.05*np.std(data*np.hamming(data.shape[0]) * (data.shape[0]-1) / np.sum(np.hamming(data.shape[0])), ddof=1)
            else:
                std[i] = np.std(data*np.hamming(data.shape[0]) * (data.shape[0]-1) / np.sum(np.hamming(data.shape[0])), ddof=1)

        h_est = 0.086 * std + 0.01 # Approximate formula for h  

        (m1, m2, m1_g), Tn = self._calc_init_estimates(y1, y2, h_est, h_est + 0.01)

        # calculate residuals
        epsilon_hat_1 = y1 - m1
        epsilon_hat_2 = y2 - m2

        """ plt.figure(dpi = 600)
        plt.plot(epsilon_hat_1)
        plt.plot(epsilon_hat_2 + 4)
        plt.legend(["epsilon 1", "epsilon 2"])
        plt.show() """

        #Tn_std = self._perform_bootstrap_var_estimation(y1, y2, m1, m2, self.h)

        #Tn = Tn / Tn_std # Studentizing

        # perform bootstrap iterations
        Tn_star = self._perform_bootstrap_iterations(epsilon_hat_1, epsilon_hat_2, m1_g)

        # evaluate test
        """ q = 1-alpha
        c_alpha_star = np.quantile(Tn_star, q) """

        self.results = {}

        for cur_alpha in alpha:
            upper_r = B
            lower_r = 0
            r = B // 2
            
            while(upper_r - lower_r > 1):
                beta_dist = stats.beta(B-r+1, r)
                if(beta_dist.cdf(cur_alpha) >= beta): # Limit search to the lower half
                    upper_r = r
                else: # Limit search to the upper half
                    lower_r = r
                r = (upper_r + lower_r) // 2
            c_alpha_star = np.sort(Tn_star)[r]
            rejected_bool = Tn > c_alpha_star
            self.results["c_alpha_star_%.5f"%cur_alpha] = c_alpha_star
            self.results["rejected_%.5f"%cur_alpha] = rejected_bool
            if printout:
                print("The Hypothesis H0 was %srejected for alpha = %.5f" %("" if rejected_bool else "not ", cur_alpha))
                print("c_alpha_star is %.5f for alpha = %.5f"%(c_alpha_star, cur_alpha))

        """ self.results = {"rejected": rejected_bool,
                        "c_alpha_star":c_alpha_star,
                        "Tn_star": Tn_star,
                        "Tn":Tn} """
        
        self.results["Tn_star"] = Tn_star
        self.results["Tn"] = Tn

        return self.results

    def plot_kde(self, title="Bootstrap Approximation", alpha=0.05):
        if self.results:
            Tn_star = self.results["Tn_star"]
            Tn = self.results["Tn"]
            c_alpha_star = self.results["c_alpha_star_%.5f"%alpha]

            fig, ax = plt.subplots()
            ax = sns.kdeplot(Tn_star, ax=ax, label="Bootstrap Distribution")
            ax.vlines(Tn, ymin=0, ymax=.6, linestyles="dashed", colors="orange", label=r"$T_n$")
            ax.vlines(c_alpha_star, ymin=0, ymax=.6, linestyles="dashed", colors="green", label=r"$c_{alpha*}$")
            ax.set_xlabel(r"$T_n$")
            ax.set_title(title)
            ax.legend()
            plt.show()

    def test_image(self, image, image_hat, h=(.02, .02), g=(.03, .03), B=1000, alpha=.05):
        """
        compare two-dimensional images to find defects

        args:
            image (array): reference image for comparison, no defects
            image_hat (array): image with possible defects
            h (tuple):      the kernel bandwidth of the initial estimates (row, col)
            g (tuple):      the kernel bandwidth of bootstrap estimates (row, col)
            B (int):        number of Bootstrap iterations
            alpha (float):  the significance level

        returns:
            defect (bool): True, if null hypothesis was rejected (defect), or False if not (no defect)
            min_point (int, int): location of minimum rejection point (row, column)
            max_point (int, int): location of maximum rejection point (row, column)
        """
        if (image.shape[1] != image_hat.shape[1]) & (image.shape[0] != image_hat.shape[0]):
            raise ValueError("Image shapes do not match.")

        def _row_iteration(i):
            return self.compute(image[i, :], image_hat[i, :], h=h[0], g=g[0], B=B, alpha=alpha, printout=False, show_progress=False)

        def _col_iteration(i):
            return self.compute(image[:, i], image_hat[:, i], h=h[1], g=g[1], B=B, alpha=alpha, printout=False, show_progress=False)
        
        # create multiprocessing pool
        pool = Pool(mp.cpu_count() - 1)

        # process rows parallel
        r = np.array([
            result["rejected"] for result in
            tqdm(pool.imap(_row_iteration, range(image.shape[0])), desc="Processing Rows", leave=True)
        ])

        # process columns parallel
        c = np.array([
            result["rejected"] for result in
            tqdm(pool.imap(_col_iteration, range(image.shape[1])), desc="Processing Columns", leave=True)
        ])


        defect_detected = any(c) or any(r)
        print(r, c)
        min_point = (np.argmax(r), np.argmax(c))
        max_point = ((len(r) - 1 - np.argmax(r[::-1])) if any(r) else 0, (len(c) - 1 - np.argmax(c[::-1])) if any(c) else 0)
        return defect_detected, min_point, max_point

    def get_results(self):
        """ returns the computed results if available """
        if self.results:
            return results
        return None

    def _resample_residual(self, epsilon_hat, B):
        """ 
        PLEASE INPUT VECTOR
        performs a classical bootstrap resample with replacement of the input array
        """
        return np.random.choice(epsilon_hat, (B, epsilon_hat.shape[0]), replace=True)

    def _sample_wild_residual(self, epsilon_hat, B):
        """ 
        PLEASE INPUT VECTOR
        returns wild residuals according to binary distribution
        """
        p = np.random.random_sample((B, *epsilon_hat.shape))
        gamma = (5 + np.sqrt(5)) / 10
        mask = p < gamma
        a = ((1-np.sqrt(5)) / 2) * np.tile(epsilon_hat,(B,*[1 for _ in range(len(epsilon_hat.shape))]))
        b = ((1+np.sqrt(5)) / 2) * np.tile(epsilon_hat,(B,*[1 for _ in range(len(epsilon_hat.shape))]))
        b[mask] = a[mask]
        return b

    def _sample_wild_residual_normal(self, epsilon_hat, B):
        """ 
        PLEASE INPUT VECTOR
        returns wild residuals according to normal distribution
        """
        V_i = np.random.standard_normal((B, epsilon_hat.shape[0]))
        return ((1/np.sqrt(2))*V_i + (1/2)*(np.square(V_i)-1)) * np.tile(epsilon_hat,(B,1))

    def _calc_init_estimates(self, y1, y2, h, g):
        # calculate initial estimates
        m1 = calc_smoothed_estimate(y1, self.kernel_function, h)
        m2 = calc_smoothed_estimate(y2, self.kernel_function, h)
        m1_g = calc_smoothed_estimate(y1, self.kernel_function, g)

        Tn = calc_Tn(m1, m2, h)

        return (m1, m2, m1_g), Tn

    def _perform_bootstrap_var_estimation(self, y1, y2, m1, m2, smoothing_coeff):
        """
        Computes one Bootstrap iteration
        1. computing wild residuals
        2. computing resulting bootstrap observations y*
        3. estimating bootstrap smoothed estimates m*
        4. calculate bootstrap test statstistic Tn*
        """

        epsilon_hat_1 = y1 - m1
        epsilon_hat_2 = y2 - m2

        bootstrap_epsilon_1 = self.residual_function(epsilon_hat_1, self.B_std)
        bootstrap_epsilon_2 = self.residual_function(epsilon_hat_2, self.B_std)

        """ plt.figure(dpi = 600)
        plt.plot(bootstrap_epsilon_1)
        plt.plot(bootstrap_epsilon_2 + 4)
        plt.legend(["epsilon 1, epsilon 2"])
        plt.show()
 """
        ndim = bootstrap_epsilon_1.ndim-1

        y1_star = np.tile(m1,(self.B_std,*[1 for _ in range(ndim)])) + bootstrap_epsilon_1
        y2_star = np.tile(m2,(self.B_std,*[1 for _ in range(ndim)])) + bootstrap_epsilon_2

        m1_star = calc_smoothed_estimate_parallel(y1_star, self.kernel_function, smoothing_coeff)
        m2_star = calc_smoothed_estimate_parallel(y2_star, self.kernel_function, smoothing_coeff)

        Tn_star = calc_Tn(m1_star, m2_star, smoothing_coeff, axis=-1)

        return np.std(Tn_star, ddof=1, axis=0)

    def _perform_bootstrap_iterations(self, epsilon_hat_1, epsilon_hat_2, m1_g):
        """
        Computes one Bootstrap iteration
        1. computing wild residuals
        2. computing resulting bootstrap observations y*
        3. estimating bootstrap smoothed estimates m*
        4. calculate bootstrap test statstistic Tn*
        """
        bootstrap_epsilon_1 = self.residual_function(epsilon_hat_1, self.B)
        bootstrap_epsilon_2 = self.residual_function(epsilon_hat_2, self.B)

        """ plt.figure(dpi = 600)
        plt.plot(bootstrap_epsilon_1[2, :])
        plt.plot(bootstrap_epsilon_2[2, :] + 4)
        plt.legend(["epsilon 1, epsilon 2"])
        plt.show() """

        y1_star = np.tile(m1_g,(self.B,1)) + bootstrap_epsilon_1
        y2_star = np.tile(m1_g,(self.B,1)) + bootstrap_epsilon_2

        m1_star = calc_smoothed_estimate_parallel(y1_star, self.kernel_function, self.h)
        m2_star = calc_smoothed_estimate_parallel(y2_star, self.kernel_function, self.h)

        bootstrap_epsilon_hat_1 = y1_star - m1_star

        std = np.zeros_like(bootstrap_epsilon_hat_1)
        N = bootstrap_epsilon_hat_1.shape[-1]
        for i in range(N):
            data = bootstrap_epsilon_hat_1[..., max(0,i-N//10):i+N//10]
            if(i > 0):
                std[..., i] = 0.95*std[..., i-1]+ 0.05*np.std(data*np.hamming(data.shape[-1]) * (data.shape[-1]-1) / np.sum(np.hamming(data.shape[-1])), ddof=1, axis=-1)
            else:
                std[..., i] = np.std(data*np.hamming(data.shape[-1]) * (data.shape[-1]-1) / np.sum(np.hamming(data.shape[-1])), ddof=1, axis=-1)

        h_est = 0.086 * std + 0.01 # Approximate formula for h 

        m1_star = calc_smoothed_estimate_parallel(y1_star, self.kernel_function, h_est)
        m2_star = calc_smoothed_estimate_parallel(y2_star, self.kernel_function, h_est)
        #h_est = self.h

        #std = self._perform_bootstrap_var_estimation(y1_star, y2_star, m1_star, m2_star, self.h)
        #print(std)
        std = 1

        Tn_star = calc_Tn(m1_star, m2_star, h_est, axis=1) # Tn_star

        return Tn_star / std


class MonteCarlo():
    """Generate Monte Carlo estimate of "true" density of Tn"""
    def __init__(self, kernel_function="bartlett_priestley"):
        # set the kernel function
        if kernel_function in ["bartlett_priestley", "bartlett_priestley_kernel", "bartlett"]:
            self.kernel_function = bartlett_priestley_kernel
        elif kernel_function in ["gaussian", "gaussian_kernel"]:
            self.kernel_function = gaussian_kernel
        else:
            raise ValueError("Unknown kernel function.")

    def compute_Tn(self, h=.02, M=1000, printout=True):
        self.h = h
        self.M = M
        pool = Pool(mp.cpu_count() - 1)
        return np.array([result for result in tqdm(pool.imap(self._sampling_iteration, np.arange(M)), total=M, leave=True)])

    def _sampling_iteration(self, m):
        # auxiliary function for each iteration
        np.random.seed((m * int(time.time())) % 123456789)
        y1, y2 = generate_data_franke(defect=False)
        _m1 = calc_smoothed_estimate(y1, self.kernel_function, self.h)
        _m2 = calc_smoothed_estimate(y2, self.kernel_function, self.h)
        return calc_Tn(_m1, _m2, self.h)

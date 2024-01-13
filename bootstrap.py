import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
        elif method in ["wild", "wild_binary"]:
            self.residual_function = self._sample_wild_residual
        elif method in ["wild_normal"]:
            self.residual_function = self._sample_wild_residual_normal
        else:
            raise ValueError("Unknown method.")

    def compute(self, y1, y2, h=.02, g=.03, B=1000, alpha=.05, printout=True, show_progress = True):
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
        self.printout = printout
        self.show_progress = show_progress

        # compute initial estimates
        (m1, m2, m1_g), Tn = self._calc_init_estimates(y1, y2)

    # calculate residuals
        epsilon_hat_1 = y1 - m1
        epsilon_hat_2 = y2 - m2

        # perform bootstrap iterations
        Tn_star = self._perform_bootstrap_iteration(epsilon_hat_1, epsilon_hat_2, m1_g)

        # evaluate test
        q = 1-alpha
        c_alpha_star = np.quantile(Tn_star, q)
        rejected_bool = Tn > c_alpha_star

        if printout:
            print("The Hypothesis H0 was %srejected" %("" if rejected_bool else "not "))
            print("c_alpha_star is %.4f"%c_alpha_star)

        self.results = {"rejected": Tn > c_alpha_star,
                        "c_alpha_star":c_alpha_star,
                        "Tn_star": Tn_star,
                        "Tn":Tn}

        return self.results

    def plot_kde(self, title="Bootstrap Approximation"):
        if self.results:
            _, c_alpha_star, Tn_star, Tn = self.results.values()

            fig, ax = plt.subplots()
            ax = sns.kdeplot(Tn_star, ax=ax, label="Bootstrap Distribution")
            ax.vlines(Tn, ymin=0, ymax=.6, linestyles="dashed", colors="orange", label=r"$T_n$")
            ax.vlines(c_alpha_star, ymin=0, ymax=.6, linestyles="dashed", colors="green", label=r"$c_{alpha*}$")
            ax.set_xlabel(r"$T_n$")
            ax.set_title(title)
            ax.legend()
            plt.show()

    def test_image(self, image, image_hat, h=.02, g=.03, B=1000, alpha=.05):
        """
        compare two-dimensional images to find defects

        args:
            image (array): reference image for comparison, no defects
            image_hat (array): image with possible defects
            h (float):      the kernel bandwidth of the initial estimates
            g (float):      the kernel bandwidth of bootstrap estimates
            B (int):        number of Bootstrap iterations
            alpha (float):  the significance level

        returns:
            defect (bool): True, if null hypothesis was rejected (defect), or False if not (no defect)
            min_point (int, int): location of minimum rejection point (row, column)
            max_point (int, int): location of maximum rejection point (row, column)
        """
        if (image.shape[1] != image_hat.shape[1]) & (image.shape[0] != image_hat.shape[0]):
            raise ValueError("Image shapes do not match.")

        results_cols = [
            self.compute(image[:, i], image_hat[:, i], h=h, g=g, B=B, alpha=alpha, printout=False, show_progress=False)
            for i in tqdm(range(image.shape[1]), desc="Processing Columns", leave=True)
        ]

        results_rows = [
            self.compute(image[i, :], image_hat[i, :], h=h, g=g, B=B, alpha=alpha, printout=False, show_progress=False)
            for i in tqdm(range(image.shape[0]), desc="Processing Rows", leave=True)
        ]
        c = [result["rejected"] for result in results_cols]
        r = [result["rejected"] for result in results_rows]
        defect = any(c) or any(r)
        min_point = (np.argmax(r), np.argmax(c))
        max_point = ((len(r) - 1 - np.argmax(r[::-1])) if any(r) else 0, (len(c) - 1 - np.argmax(c[::-1])) if any(c) else 0)
        return defect, min_point, max_point

    def results(self):
        """ returns the computed results if available """
        if self.results:
            return results
        return None

    def _resample_residual(self, epsilon_hat):
        """ 
        PLEASE INPUT VECTOR
        performs a classical bootstrap resample with replacement of the input array
        """
        return np.random.choice(epsilon_hat, epsilon_hat.shape[0], replace=True)

    def _sample_wild_residual(self, epsilon_hat):
        """ 
        PLEASE INPUT VECTOR
        returns wild residuals according to binary distribution
        """
        p = np.random.random_sample(epsilon_hat.shape[0])
        gamma = (5 + np.sqrt(5)) / 10
        mask = p < gamma
        a = ((1-np.sqrt(5)) / 2) * epsilon_hat
        b = ((1+np.sqrt(5)) / 2) * epsilon_hat
        b[mask] = a[mask]
        return b

    def _sample_wild_residual_normal(self, epsilon_hat):
        """ 
        PLEASE INPUT VECTOR
        returns wild residuals according to normal distribution
        """
        V_i = np.random.standard_normal(epsilon_hat.shape[0])
        return ((1/np.sqrt(2))*V_i + (1/2)*(np.square(V_i)-1))*epsilon_hat

    def _calc_init_estimates(self, y1, y2):
        # calculate initial estimates
        m1 = calc_smoothed_estimate(y1, self.kernel_function, self.h)
        m2 = calc_smoothed_estimate(y2, self.kernel_function, self.h)
        m1_g = calc_smoothed_estimate(y1, self.kernel_function, self.g)

        Tn = calc_Tn(m1, m2, self.h)
        return (m1, m2, m1_g), Tn

    def _bootstrap_iteration(self, b, epsilon_hat_1, epsilon_hat_2, m1_g):
        """
        Computes one Bootstrap iteration
        1. computing wild residuals
        2. computing resulting bootstrap observations y*
        3. estimating bootstrap smoothed estimates m*
        4. calculate bootstrap test statstistic Tn*
        """
        np.random.seed((b * int(time.time())) % 123456789) # setting seed so that every iteration uses different random number generator starting points
        bootstrap_epsilon_1 = self.residual_function(epsilon_hat_1)
        bootstrap_epsilon_2 = self.residual_function(epsilon_hat_2)

        y1_star = m1_g + bootstrap_epsilon_1
        y2_star = m1_g + bootstrap_epsilon_2

        m1_star = calc_smoothed_estimate(y1_star, self.kernel_function, self.h)
        m2_star = calc_smoothed_estimate(y2_star, self.kernel_function, self.h)

        return calc_Tn(m1_star, m2_star, self.h) # Tn_star

    def _perform_bootstrap_iteration(self, epsilon_hat_1, epsilon_hat_2, m1_g):
        # Parallel processing of one boostrap iteration
        Tn_star = []
        pool = Pool(mp.cpu_count()-1)
        # create auxiliary function
        func = functools.partial(self._bootstrap_iteration,
                                 epsilon_hat_1=epsilon_hat_1,
                                 epsilon_hat_2=epsilon_hat_2,
                                 m1_g=m1_g)
        # collect results of B iterations

        if self.show_progress:
            results = [result for result in tqdm(pool.imap(func, np.arange(self.B)), total=self.B, leave=True)]
        else:
            results = pool.map(func, np.arange(self.B))
        Tn_star = np.array(results)
        return Tn_star


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

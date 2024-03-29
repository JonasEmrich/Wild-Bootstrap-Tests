from tqdm import tqdm
import time
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd

from utils import *
from bootstrap import *

class MonteCarloEvaluation():
    '''
    This class implements a simple evaluation setup that performs montecarlo trials of a given function based on data provided using a data generating function
    '''
    def __init__(self, data_generator, testing_method):
        """
        data_generator has to be a function in the format: data = F()
        testing_method has to be a function in the format: [bool] = T(data)
        """
        self._data_generator = data_generator
        self._testing_method = testing_method

    def perform_trials(self, N=500, filename=None, name=""):
        """
            Starts the excecution of MC experments
            Saves results when a filename is provided under the provided name

        """
        self.filename = filename
        self.name = name

        # Perform L monte carlo experiments/bootstrap hypothesis tests

        pool = Pool(mp.cpu_count()-1)
        results = np.array([result for result in tqdm(pool.imap(self._iter, np.arange(N)), total=N, leave=True)])

        # write
        if filename is not None:
            df = pd.DataFrame(results)
            df['name'] = name
            df.to_csv(filename, header=False, index=False, mode="a")

        return results


    def _iter(self, i):
        """ Performs one monte carlo iteration of testing a signal using Bootstrap"""
        np.random.seed((i * int(time.time())) % 123456789)

        # Generate observational data
        data = self._data_generator()

        # perform computation
        result = self._testing_method(*data)
    
        return result



if __name__ == "__main__":
    # define evaluation setup
    filename = f"data/evaluation_franke.csv"
    N = 1000 # number of runs   
    methods = ["normal", "wild"]
    defects = [False, True]
    noise_distributions = ["gauss","skewed", "laplace"]

    for method in methods:
        for noise in noise_distributions:
            for defect in defects:
                BS = Bootstrap(method=method, kernel_function="bartlett_priestley_kernel")
                MC = MonteCarloEvaluation(data_generator = lambda: generate_data_franke(defect=defect, noise=noise),
                                         testing_method = lambda y1, y2: BS.compute(y1, y2, h=.02, g=.03, B=1000, alpha=.05, printout=False)["rejected"])

                # start MC computation
                name = method+"_"+noise+"_defected_data" if defect else method+"_"+noise+"_typical_data"

                print(f"Starting Evaluation with N={N}, method={method}, noise={noise}, defect={defect}")
                MC.perform_trials(N=N, filename=filename, name=name)

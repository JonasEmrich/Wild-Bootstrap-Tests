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
        np.random.seed((i * int(time.time()-begin_time)) % 123456789)

        # Generate observational data
        data = self._data_generator()

        # perform computation
        result = self._testing_method(*data)
    
        return result


defect = True
method = "normal" # "wild"
BS = Bootstrap(method=method, kernel_function="bartlett_priestley_kernel")
begin_time = time.time()

if __name__ == "__main__":
    # define evaluation setup
    filename = f"data/evaluation_heavyside03_noise_h07.csv"
    N = 250 # number of runs   
    # method = "normal"
    """     defect = True


    BS = Bootstrap(method=method, kernel_function="bartlett_priestley_kernel") """
    MC = MonteCarloEvaluation(data_generator = lambda: generate_data_franke(defect=defect),
                             testing_method = lambda y1, y2: BS.compute(y1, y2, h=.07, g=.08, B=1000, B_std=25, alpha=.05, beta=0.95, printout=False)["rejected"])

    # start MC computation
    name = method+"_defected_data" if defect else method+"_typical_data"
    print(f"Starting Evaluation with N={N}, method={method}, defect={defect}")
    MC.perform_trials(N=N, filename=filename, name=name)

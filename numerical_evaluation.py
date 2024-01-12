from tqdm import tqdm
import time

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

        # Perform L monte carlo experiments/bootstrap hypothesis tests
        results = np.zeros(N)
        for i in tqdm(range(N)):
            results[i] = self._iter(i)

            if filename is not None:
                with open(filename, "a") as file:
                    file.write(f"{name}, {results[i]} \n")
        return results


    def _iter(self, i):
        """ Performs one monte carlo iteration of testing a signal using Bootstrap"""
        np.random.seed((i * int(time.time())) % 123456789)

        # Generate observational data
        data = self._data_generator()

        # perform computation
        result = self._testing_method(*data)
    
        return result





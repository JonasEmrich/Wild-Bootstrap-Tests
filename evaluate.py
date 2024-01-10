from tqdm import tqdm
import time

from utils import *
from bootstrap import *

def iter(i):
    """ Performs one monte carlo iteration of testing a signal using Bootstrap"""
    np.random.seed((i * int(time.time())) % 123456789)

    # Generate observational data
    y1, y2 = generate_data_franke(defect=defect)

    BS = Bootstrap(method="wild", kernel_function="bartlett_priestley_kernel")
    results_wild = BS.compute(y1, y2, h=.02, g=.03, B=1000, alpha=.05, printout=False)
    #BS.plot_kde(title="Bootstrap Approximation Wild")

    return results_wild["rejected"]

# evaluation setup
L = 50 # number of tests
defect = True # testing with a defected signal
filename = f"evaluation_defect={defect}.csv"

# Perform L monte carlo experiments/bootstrap hypothesis tests
DATA = np.zeros(L)
for i in tqdm(range(L)):
    DATA[i] = iter(i)

    with open(filename, "a") as file:
        file.write(str(DATA[i])+"\n")

# Read Data
DATA = np.genfromtxt(filename, delimiter=',')
FAR = len(DATA[DATA==defect])/len(DATA)
print(f"False Alarm Rate {FAR}")
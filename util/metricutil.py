from matplotlib import pyplot as plt, axes as axes
import numpy as np
import pandas as pd
import scipy as sp
import math
import logging
from itertools import product
from numba import jit, njit, prange

from ceciestunepipe.util.sound import temporal as tu

logger = logging.getLogger('ceciestunepipe.util.metricutil')

def pair_dist(x, y, dist_fun, fun_args:tuple=(), fun_kwargs: dict={}, ref_long=False):
    a, b = tu.iso_len_pair(x, y, ref_long=False)
    
    return dist_fun(a.flatten(), b.flatten(), *fun_args, **fun_kwargs)

def scalar_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def cos_sim(x, y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))
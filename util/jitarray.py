from numba import jit
import numpy as np


@jit(nopython=True, parallel=True)
def substract_mean_col(x: np.array) -> np.ndarray:
    """ remove offset of a 2d array along axis 1
    Arguments:
        x {np.array} -- [n_i, n_j] array
    
    Returns:
        np.ndarray -- [n_i, n_j] array, with the mean of each column substracted along it
    """
    # remove offset
    n_chan, n_sample = x.shape
    y = np.empty(x.shape)
    for i in np.arange(n_chan):
        y[i, :] = x[i, :] - np.mean(x[i, :])
    return y

@jit(nopython=True, parallel=True)
def mean_2d(x, axis=1):
    #mean along axis 1
    
    n0, n1 = x.shape

    y
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)
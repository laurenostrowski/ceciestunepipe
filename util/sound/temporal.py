import numpy as np
import logging

logger = logging.getLogger('ceciestunepipe.util.sound.temporal')

def vector_1darray(func):
    # decorates functions so they can work with vectors as well as arrays
    # takes x, expands dimension, applies function, squeezes dimensions
    # only applies to function returning array
    def wrapper(x, *args, **kwargs):
        #logger.info('wrapping in')
        #logger.info('xdim {}'.format(x.ndim))
        if x.ndim==1:
            y = np.expand_dims(x, 0)
            z = func(y, *args, **kwargs).squeeze()
        else:
            z = func(x, *args, **kwargs)
        return z
    return wrapper

@vector_1darray
def resample_interp(y: np.array, n: int) -> np.array:
    # resample interpolate using np.interp, along the last index of the array
    n_samples = y.shape[-1]
    # both coordiante vectors [0, 1]
    t = np.arange(n_samples)/n_samples
    t_r = np.arange(n)/n
    y_r = np.stack([np.interp(t_r, t, y_row) for y_row in y], axis=0)
    return y_r

def iso_len_pair(x: np.array, y: np.array, ref_long: bool=False) -> tuple:
    # returns both arrays, one of them resampled to the ref_size, along the last axis
    # ref_short: whether to use the short one or the long one as a reference.
    arr_list = [x, y]
    len_arr = np.array([x.shape[-1] for x in arr_list])
    # get their length order so that the first is the ref
    len_order = np.argsort(len_arr)
    if ref_long:
        len_order = len_order[::-1]
    
    # resample the second one to the length of the first one
    i_ref, i_res = len_order
    arr_list[i_res] = resample_interp(arr_list[i_res], len_arr[i_ref])
    return arr_list

def rms(x:np.array, axis: int=0) -> np.array:
    #print('plain rms with shape {}'.format(x.shape))
    if axis is None:
        return np.linalg.norm(x)
    else:
        return np.linalg.norm(x, axis=axis)/np.sqrt(x.shape[axis])

def quick_env(x: np.array, avg_bin: int) -> np.array:
    # envelope of a 1 dimension array
    x_env = np.convolve((np.abs(x)), np.ones(avg_bin), mode='same')
    return x_env

def fake_pressure(x, avg_bin: int) -> np.array:
    #fake pressure signal from sound trace
    x_env = quick_env(x, avg_bin)
    return np.log(x_env)





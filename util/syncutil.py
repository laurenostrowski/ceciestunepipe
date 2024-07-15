import os
import sys
import glob
import logging
import datetime
import parse
import shutil
import copy
import numpy as np
import pandas as pd
import warnings
import pickle

from scipy.interpolate import interp1d

logger = logging.getLogger('ceciestunepipe.util.syncutil')

def square_to_edges(x: np.array) -> np.array:
        
        dig = np.squeeze(x)
        diff_dig = np.diff(x)

        rising = np.where(diff_dig > 0)[0]
        falling = np.where(diff_dig < 0)[0]

        ttl_frames = np.concatenate((rising, falling))
        ttl_states = np.array([1] * len(rising) + [-1] * len(falling))
        sort_idxs = np.argsort(ttl_frames)

        return ttl_frames[sort_idxs], ttl_states[sort_idxs]

def quick_ttl_threshold(x:np.array) -> float:
    # assumes values oscilate roughly between two points (no outlier noise)
    thresh = np.min(x) + np.ptp(x)/2
    return thresh


def sync_to_pattern(x_ttl: np.array, t: np.array, x_0_ttl: np.array, t_0:np.array) -> np.array:
    # x_ttl is the array of [2, n_transition] ([transition_sign, sample] (transition = +/-1))
    # strategy is
    # check:
        # first edges are both same sign 
        # number of edges is the same
    # get times to interpolate:
    # get the 'actual' times at the edges, i.e the t_0 at the locationn of the edges in x_0, regardless of the sign (two values per cycle, means T/2 is the actual period)
    # make an interpolation function with it
    # fit the t to those values (we know that t=t_0 at those edges, the rest is the correciton of the interpolation)
    
    n_edges = x_ttl.shape[1]
    n_edges_0 = x_0_ttl.shape[1]
    if x_ttl[1, 0] != x_0_ttl[1, 0]:
        # If the signals don't have the same number of edges there may be an error, better stop and debug
        raise ValueError(
            'Sign of first edge transition of pattern and target dont match')
    if n_edges != n_edges_0:
        # If the signals don't have the same number of edges there may be an error, better stop and debug
        raise ValueError(
            'Number of edges in the syn ttl events of pattern and target dont match')
        
    # if all checks out, do the deed
    t_0_edge = t_0[x_0_ttl[0]]
    sample_edge = x_ttl[0]
    # the interpolation function. fill_value='extrapolate' allows extrapolation from zero and until the last time stamp
    # careful, this could lead to negative time, but it is the correct way to do it.
    # interpolation function interpolates time as a target, t_0=f(sample) with true values at the edges
    t_interp_f = interp1d(sample_edge, t_0_edge,
                          assume_sorted=True, 
                          fill_value='extrapolate')
    
    n_samples = t.size
    t_prime = t_interp_f(np.arange(n_samples))
    return t_prime



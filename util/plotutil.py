from matplotlib import pyplot as plt, axes as axes
import numpy as np
import scipy as sp
import math
from numba import jit, njit, prange
import scipy

axes_pars = {'axes.labelpad': 5,
             'axes.titlepad': 5,
             'axes.titlesize': 'small',
             'axes.grid': False,
             'axes.xmargin': 0,
             'axes.ymargin': 0}

plt.rcParams.update(axes_pars)

# fucntions for handling and plotting
def decim(x, q):
    # decimate a 1 x n array
    # x: 1xn matrix (float)
    # q: int (decimate ratio), 0<q<=x.size
    assert (x.size >= q and q > 0)
    pad_size = int(math.ceil(float(x.size) / q) * q - x.size)
    pad = np.empty(pad_size)
    pad[:] = np.nan
    x_padded = np.append(x, pad)
    return np.nanmean(x_padded.reshape(-1, q), axis=1)


# bin the array in columns
def col_binned(a, bs):
    # a: rectangular array shape (n, m)
    # bs: size for bin columns
    # output : array (n, o=m//bs)
    # if m<o*bs, those columns are padded with zeros

    n = a.shape[0]
    m = a.shape[1]
    o = int(np.ceil(m / bs))

    pad = np.empty([n, o * bs - m]) * np.nan
    padded = np.append(a, pad, axis=1)
    return np.nansum(padded.reshape(n, o, bs), axis=2)

# bin the array in columns
def col_binned_max(a, bs):
    # a: rectangular array shape (n, m)
    # bs: size for bin columns
    # output : array (n, o=m//bs)
    # if m<o*bs, those columns are padded with zeros

    n = a.shape[0]
    m = a.shape[1]
    o = int(np.ceil(m / bs))

    pad = np.empty([n, o * bs - m]) * np.nan
    padded = np.append(a, pad, axis=1)
    return np.max(padded.reshape(n, o, bs), axis=2)

def coarse(x: np.ndarray, n_coarse: int):
    # coarse x along last dimension in bins of n_coarse size
    x_shape = x.shape
    last_dim = x.shape[-1]
    
    max_n = last_dim//n_coarse * n_coarse # chop it to the max integer coarse
    exploded_x = x[..., :max_n].reshape(*x_shape[:-1], -1, n_coarse)
    coarse_x = np.nanmean(exploded_x, axis=-1)
    return coarse_x

def plot_as_raster(x, ax=None, t_0=None, t_f=None):
    #x is [n_events, n_timestamps] array
    n_y, n_t = x.shape
    x[x==0] = np.nan
    
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlim(0, n_t)
        ax.set_ylim(0, n_y + 1)
        
    x_trial, x_time = np.where(x>0)
    raster = ax.plot(x_time, x_trial, '.', color='k', markersize=0.75, rasterized=True)
    
    if t_0 is not None:
        ax.axvline(x=t_0, color='red')

    if t_f is not None:
        ax.axvline(x=t_f, color='red')
    return ax
    
def plot_trial_raster(trial_raster: np.array, ev_marks: dict={}, ax=None, bin_size=0):
    if ax is None:
        fig, ax = plt.subplots()

    if bin_size == 0: # plain raster
        ax.plot(trial_raster, np.arange(trial_raster.shape[0]), 'k|')
        ax.yaxis.set_ticks([0, trial_raster.shape[0]])
        ax.set_ylim(0, trial_raster.shape[0]*1.1)
        ax.set_ylabel('trial #')
        time_scale = 1
    else: # gte the psth
        raster_sparse = sparse_raster(trial_raster)
        psth = col_binned(raster_sparse, bin_size).mean(axis=0)*1000/bin_size
        time_scale = bin_size
        ax.plot(np.arange(psth.size)*time_scale, psth, 'r')
        ax.yaxis.set_ticks([int(np.max(psth))])
        ax.set_ylim([0, (np.max(psth)*1.1)])
        ax.set_ylabel('F.R. (Hz)')
        
    ax.set_xlim(0, trial_raster.shape[1])
    for ev, mark in ev_marks.items():
        ax.axvline(x=mark,)
    return ax


def plot_raster(x, t1=0, t2=-1, t0=0, ax=None, bin_size=0):
    # plot a raster
    # x: spikes matrix:
    # nxt matrix with 1 where there is a spikes.
    # cols: time stamps (ms)
    # rows: trials

    # t1 from beggining of x to plot: default 0, dont cut
    # t2 time after begginning of x to plot: default -1, all range
    # t0 where to put the 0 (stimulus mark) relative to the range t1:t2
    # ax: axes object where to put the plot in (default = None, create a new one)
    # bin_size: int

    # Returns:
    # raster: a PathCollection (if bin_size=0) or a Line2D object (if bin_size=1)
    # ax    : Axes object

    # prepare the axis
    # if no axis, make a new plot
    if ax is None:
        raster_fig = plt.figure()
        ax = raster_fig.add_axes([0, 0, 1, 1])

    # pdb.set_trace()
    # if bin_size was entered, we want a psth
    if bin_size > 0:
        psth, t_dec = make_psth(x, t1=t1, t2=t2, t0=t0, bin_size=bin_size)
        raster = ax.plot(t_dec, psth)
        ax.set_ylim(0, max(psth) * 1.2)
        stim = ax.plot((0, 0), (0, max(psth) * 1.2), 'k--')
        t_max = max(t_dec)

    else:
        # Chop the segment
        if t2 > 0:
            assert (t2 > t1)
            x = x[:, t1:t2]
        else:
            x = x[:, t1:]

        # get dimensions and time
        events = x.shape[0]
        t_stamps = x.shape[1]
        t = np.arange(t_stamps) - t0

        # mask the zeros (no spike)
        nsp = x[:] == 0
        # x[nsp]=np.nan

        # make the frame for plotting
        row = np.ones(t_stamps, dtype=float)
        col = np.arange(events, dtype=float)
        frame = col[:, np.newaxis] + row[np.newaxis, :]

        raster = ax.scatter(t * x, frame * x, marker='|', rasterized=True)
        ax.set_ylim(0, events + 1)
        ax.plot((0, 0), (0, events + 1), 'k--')
        t_max = t_stamps - t0

    ax.set_xlim(0 - t0, t_max)
    return raster, ax


# make a psth from a spikes matrix
def make_psth(x, t1=0, t2=-1, t0=0, bin_size=1):
    # x: spikes matrix:
    # nxt matrix with 1 where there is a spikes.
    # cols: time stamps (ms)
    # rows: trials

    # t1 from beginning of x: default 0, dont cut
    # t2 time after beginning of x to cut: default -1, all range
    # bin_size: int

    # Returns:
    # psth: an array with the frequency (counts/(bin_size*n_trials))

    # Chop the segment
    if t2 > 0:
        assert (t2 > t1)
        x = x[:, t1:t2]
    else:
        x = x[:, t1:]

    # get dimensions and time
    events = x.shape[0]
    t_stamps = x.shape[1]
    t = np.arange(t_stamps) - t0

    # pdb.set_trace()
    # if bin_size was entered, we want a psth
    # x = x[:t_stamps, :]

    t_dec = decim(t, bin_size)
    n_bins = t_dec.shape[0]
    pad_size = n_bins * bin_size - x.shape[1]
    pad = np.zeros(pad_size, dtype=int)

    psth = np.sum(np.append(pad, np.sum(x, axis=0)).reshape(n_bins, bin_size), axis=1) / (events * bin_size * 0.001)
    return psth, t_dec


# grab a raster in format row=timestamps, col=trials
# and turn it into a matrix n_trials x t_samples with a one wherever there is a spike
def sparse_raster(x, nan=False):
    n_t = x.shape[0] # n of trials
    n_s = x.shape[1] # n of samples
    raster = np.empty_like(x)
    raster[:] = np.nan

    for trial in np.arange(n_t):
        # print(trial)
        # if(trial==15):
        #     print(np.nanmax(x[trial, :]))
        r = x[trial, :] - 1
        raster[trial, np.array(r[~np.isnan(r)], dtype=int)] = 1

    if not nan:
        raster[np.isnan(raster)] = 0
    return raster

#@njit(parallel=True)
def plottable_array(x:np.ndarray, scale:np.ndarray, offset:np.ndarray) -> np.ndarray:
    """ Rescale and offset an array for quick plotting multiple channels, along the 
        1 axis, for each jth axis
    Arguments:
        x {np.ndarray} -- [n_col x n_row] array (each col is a chan, for instance)
        scale {np.ndarray} -- [n_col] vector of scales (typically the ptp values of each row)
        offset {np.ndarray} -- [n_col] vector offsets (typycally range (row))

    Returns:
        np.ndarray -- [n_row x n_col] scaled, offsetted array to plot
    """
    # for each row [i]:
    # - divide by scale_i
    # - add offset_i
    n_row, n_col = x.shape
    for col in prange(n_col):
        col_mean = np.mean(x[:, col])
        for row in range(n_row):
            x[row, col] = (x[row, col] - col_mean)* scale[col] + offset[col]
    return x


def simple_plottable_arr(x: np.ndarray, scale: str='max')-> np.array:
    """ Rescale and offset an array for quick plotting multiple channels, along the 
        1 axis, for each axis 0
    Arguments:
        x {np.ndarray} -- [n_t x n_chan] array (each row is a sample, col is a chan, for instance)
    Returns:
        np.ndarray -- [n_t x n_chan] scaled, offsetted array to plot
    """
    # get the ptp for each channel
    y = scipy.stats.zscore(x, axis=0)
    # traspose so channels are zero axis
    if scale=='each':
        y_scaled = np.vstack([z/np.ptp(z) for z in y.T]).T
    elif scale=='max':
        y_scaled = y/np.max(np.ptp(y, axis=0))
    
    y_offset = y_scaled + np.arange(x.shape[-1]) + 0.5
    return y_offset


def plot_array(x: np.ndarray, scale='each', ax=None, offset_scale=1) -> axes.Axes:

    """ Rescale and offset an array for quick plotting multiple channels, along the 
        1 axis, for each jth axis
    Arguments:
        x {np.ndarray} -- [n_col x n_row] array (each col is a chan, for instance)
    
    Keyword Arguments:
        scale {str} -- {'each', 'max'} (default: {'each'}) whether to scale within each col
                        or to the max ptp of all cols
        ax {[type]} -- [description] (default: {None})
    
    Returns:
        axes.Axes -- [description]
    """
    if ax is None:
        _, ax = plt.subplots()
    
    # arrange the array:
    n_row, n_col = x.shape
    offset = np.arange(n_col) * offset_scale
    ptp = np.ptp(x, axis=0)
    if scale == 'max':
        ptp[:] = np.max(ptp)
    
    x_scaled = plottable_array(x, 1./ptp, offset)
    ax.plot(x_scaled)

    






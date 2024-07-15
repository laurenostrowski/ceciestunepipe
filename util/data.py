import numpy as np

def rolling_window(a:np.ndarray, window: int, step:int=1)-> np.ndarray:
    """
    Create a rolled view of the array with window size, along the last dimension.
    The view explodes the window in the last dimension of the array, thus adds one dimension.
    Parameters
    ----------
    a: ndarray, shape=(n_0, n_1, ...n_N, n_t)
        Input array, to roll window along the last dimension (n_t)
    window: int
        Window size
    step: int
        Stride for the window to be rolled
    Returns
    -------
    a_rolled : np.ndarray, shape=(n_0, n_1, ...n_N, n_t//step, window)
        N+1 dimension array, with the last dimension being the window length.
    
    Note that if step=window, it is the equivalent to a simple reshape
    """
    
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    a_rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolled[..., ::step, :] # skip steps along the (n-1)th axis

def feature_flat(x: np.ndarray) -> np.ndarray:
    """
    Reshape a rolled window to a 2d array with flattened windows.
    Parameters
    ----------
    x: ndarray, shape=(n_features, n_samples, win_size)
        Input array, result of window-roll an array with rolling_window
    Returns
    -------
    x_flat_features : np.ndarray, shape=(n_features*window, n_samples)
        Array with the rolled windows flattened and placed in the first index
    """
    n_samples = x.shape[1]
    return x.transpose(0, 2, 1).reshape(-1, n_samples)


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    taken from Tim Sainburg or Marvin Theilk
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    x = np.hstack((X, append))

    valid = len(x) - window_size
    nw = valid // window_step
    out = np.ndarray((nw, window_size), dtype=x.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * window_step
        stop = start + window_size
        out[i] = x[start: stop]
    return out


def pad_spectrogram(x: np.array, pad_length: int) -> np.array:
    """ Pads a spectrogram to being a certain length along the last axis
    """
    excess_needed = pad_length - np.shape(x)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        x, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )

import numpy as np
import umap
import pandas as pd
from matplotlib import pyplot as plt

from umap.parametric_umap import ParametricUMAP

import logging
from ceciestunepipe.util import data as dt

logger = logging.getLogger('ceciestunepipe.util.dimension.umap')


def rolling_umap(x: np.ndarray, win_size: int, step: int=1, reducer=None, parametric: bool=False, 
                    only_fit: bool=False, umap_kwargs: dict={}) -> tuple:
    # roll along last dimension with step 1, window size window
    rolled_x = dt.rolling_window(x, win_size, step=step)
    #rolled_x with this transposition will be [x.shape[0], x.shape[2]//win_size, win_size] = [n_feat, n_samples, win_size]
    n_feat, n_samp, win_size = rolled_x.shape
    # want y to be [x.shape[1], x_shape[2] * x.shape[0]]
    rolled_x_featflat = rolled_x.transpose(0, 2, 1).reshape(-1, n_samp)
    
    if reducer is None:
        if parametric:
            reducer = ParametricUMAP(**umap_kwargs)
        else:
            reducer = umap.UMAP(**umap_kwargs)
        if only_fit:
            embedding = reducer.fit(rolled_x_featflat.T)
        else:
            embedding = reducer.fit_transform(rolled_x_featflat.T)
    else:
        embedding = reducer.transform(rolled_x_featflat.T)
    
    return reducer, embedding, rolled_x_featflat


def rolled_for_umap(x: np.ndarray, win_size: int, step: int=1, flatten=True) -> np.array:
    
    # roll along last dimension with step step, window size window
    rolled_x = dt.rolling_window(x, win_size, step=step)
    #rolled_x with this transposition will be [x.shape[0], x.shape[2]//win_size, win_size] = [n_feat, n_samples, win_size]
    n_feat, n_samp, win_size = rolled_x.shape
    # want y to be [x.shape[1], x_shape[2] * x.shape[0]]
    if flatten:
        rolled_x = rolled_x.transpose(0, 2, 1).reshape(-1, n_samp)
        
    return rolled_x
    

    
def df_rolled_umap(df: pd.DataFrame, data_arr_key: str, win_size: int, step: int=1, 
                     reducer=None, parametric: bool=False, 
                    only_fit: bool=False, umap_kwargs: dict={}):
    
    # gets the rolled umap projections for all the series in a dataframe using the
    # data_arr_key (rolling along the last dimension)
    # get the rolling window for all series and store it in the r_key column
    logger.info('Getting rolled umap for key {} in dataframe'.format(data_arr_key))
    logger.info('window size {}'.format(win_size))

    # get all the rolled windows and flatten the features
    r_key = '{}_roll_{}-{}'.format(data_arr_key, win_size, step)
    logger.info('getting the rolling window array for all series in the dataframe')
    df[r_key] = df[data_arr_key].apply(lambda x: rolled_for_umap(x, win_size, step))

    # concatenate all the rolled-window time series with t along the last axis
    # fit_transform
    logger.info('Instantiating a reducer and fitting-transforming into the embedding')
    reducer = umap.UMAP(**umap_kwargs)
    data_arr_cat = np.concatenate(list(df[r_key]), axis=1) # [n_feat, n_time]
    # we want the embedding to reduce on the features axis not the t so transpose
    embedding = reducer.fit_transform(data_arr_cat.T)

    # split the embedding of the concatenated array into the series, using where the start:end
    # in the rolling window concatenated array

    ### get the embeddings back into the dataframe
    umap_key = 'umap-{}_{}-{}'.format(data_arr_key, win_size, step)
    logger.info('splitting the concatenated series back into each row of the df in key {}'.format(umap_key))
    df['umap_n'] = df[r_key].apply(lambda x: x.shape[-1])
    df['umap_start'] = np.concatenate([np.array([0]), np.cumsum(df['umap_n'].values)[:-1]])
    df['umap_end'] = np.cumsum(df['umap_n']) 
    df[umap_key] = df.apply(lambda s: embedding[s['umap_start']:s['umap_end']], axis=1)

    return df, umap_key


def df_rolled_each_umap(df: pd.DataFrame, data_arr_key: str, win_size: int, step: int=1, 
                     reducer=None, parametric: bool=False, 
                    only_fit: bool=False, umap_kwargs: dict={}):
    
    # gets the rolled umap projections for all the series in a dataframe using the
    # data_arr_key (rolling along the last dimension)
    # get the rolling window for all series and store it in the r_key column
    logger.info('Getting rolled umap for key {} in dataframe'.format(data_arr_key))
    logger.info('window size {}'.format(win_size))

    if reducer is None:
        reducer = umap.UMAP(**umap_kwargs)
    # split the embedding of the concatenated array into the series, using where the start:end
    # in the rolling window concatenated array

    ### get the embeddings back into the dataframe
    umap_key = 'umap-{}_{}-{}'.format(data_arr_key, win_size, step)

    df[umap_key] = df[data_arr_key].apply(lambda x: x_rolled_umap(x, win_size, step,
                                                                  reducer=reducer,
                                                                  umap_kwargs=umap_kwargs))

    return df, umap_key

def x_rolled_umap(x: np.array, win_size: int, step: int=1, 
                     reducer=None, parametric: bool=False, 
                    only_fit: bool=False, umap_kwargs: dict={}):
    rolled_x = rolled_for_umap(x, win_size, step)
    embedding = reducer.fit_transform(rolled_x.T)
    return embedding

def ds_rolled_umap(ds, data_arr_key: str, win_size: int, step: int=1, 
                     reducer=None, parametric: bool=False, 
                    only_fit: bool=False, umap_kwargs: dict={}):
    
    rolled_x = rolled_for_umap(ds[data_arr_key], win_size, step)
    umap_key = 'umap-{}_{}-{}'.format(data_arr_key, win_size, step)

    if reducer is None:
        reducer = umap.UMAP(**umap_kwargs)

    embedding = reducer.fit_transform(rolled_x.T)
    return embedding





def plot_embedding(ds: pd.Series, data_key: str, umap_key: str,
                   ax_arr = None, plot_range: np.array = np.array([0, 1])):

    emb = ds[umap_key]
    x = ds[data_key]

    # time is relative to the bout and goes 0, 1 (for color mapping)
    t_emb = np.arange(emb.shape[0])/emb.shape[0]
    emb_range = (t_emb > plot_range[0]) & (t_emb < plot_range[1])
    
    if ax_arr is None:
        fig, ax_arr = plt.subplots(nrows=4, 
                                   gridspec_kw={'height_ratios': [4, 1, 1, 1]}, 
                                   figsize=(10, 16), 
                                   sharex=False)    


    # the plot of the embedding for the range
    ax_arr[0].scatter(emb[emb_range, 0], emb[emb_range, 1], c=t_emb[emb_range], 
                      s=2, 
                      cmap='inferno')
    ax_arr[0].set_title('{} {}'.format(data_key, umap_key))

    # the pot of the features
    ax_arr[1].scatter(t_emb[emb_range], emb[emb_range, 0], c=t_emb[emb_range], 
                      s=0.1, 
                      cmap='inferno')
    #ax_arr[1].set_xticks([])
    
    ax_arr[2].scatter(t_emb[emb_range], emb[emb_range, 1], c=t_emb[emb_range], 
                      s=0.1, 
                      cmap='inferno')
    ax_arr[2].set_xticks([])
    
    # the plot of the actual signal
    t_x = np.arange(x.shape[1])/x.shape[1]
    x_range = (t_x > plot_range[0]) & (t_x < plot_range[1])

    if 's_xx' in data_key:
        ax_arr[3].imshow(np.log(x[:, x_range][::-1]), cmap='inferno', aspect='auto')
    else:
        ax_arr[3].imshow((x[:, x_range][::-1]), cmap='inferno', aspect='auto')
    
    ax_arr[3].set_xlabel('t (ms)')

    return ax_arr




def poincare_section(time_series, section_dim, section_val, plot=True):
    """
    Perform a Poincaré section analysis on a 2-dimensional time series to find periodic orbits.

    Args:
        time_series (ndarray): The 2-dimensional time series array of shape (N, 2).
        section_dim (int): The dimension along which to create the Poincaré section (0 or 1).
        section_val (float): The value at which to create the Poincaré section.
        plot (bool): Whether to plot the Poincaré section (default: True).

    Returns:
        ndarray: The Poincaré section array of shape (M, 2), where M is the number of points on the section.

    """

    # Create empty arrays to store Poincaré section points
    poincare_points = []

    # Iterate through the time series
    for i in range(1, len(time_series)):
        prev_point = time_series[i - 1]
        curr_point = time_series[i]

        # Check if the current point crosses the Poincaré section
        if prev_point[section_dim] < section_val and curr_point[section_dim] >= section_val:
            # Interpolate the intersection point
            t = (section_val - prev_point[section_dim]) / (curr_point[section_dim] - prev_point[section_dim])
            poincare_point = prev_point + t * (curr_point - prev_point)
            poincare_points.append(poincare_point)

    poincare_points = np.array(poincare_points)

    # Plot the Poincaré section
    if plot:
        plt.scatter(poincare_points[:, 0], poincare_points[:, 1], marker='.', color='b')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Poincaré Section')
        plt.show()

    return poincare_points

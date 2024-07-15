import logging
import pandas as pd
import numpy as np

import neo
import quantities as pq
from elephant.gpfa import GPFA
import viziphant as vp

from sklearn.decomposition import PCA

logger = logging.getLogger('ceciestunepipe.util.dimension.gpfa')

def spike_trains_from_spk_arr(all_spk_arr: np.array,
                s_f: int=30000, 
                t_pre=0, t_post=0, t_stop=None,
                unit_idx_arr=None)->list:
    # get the time of the spikes
    # make a neo SpikeTrain for each cluster
    
    # filter units
    spk_arr = all_spk_arr if unit_idx_arr is None else all_spk_arr[unit_idx_arr]
    
    # if a fixed time, then a fixed time, if not make it from the trial length
    # trim the relevant segments of the array (sample is axis=1)
    pre_ap, post_ap = [int(x * s_f) for x in [t_pre, t_post]]
    
    if t_stop is None:
        x_arr = spk_arr[:, -pre_ap:-post_ap] if post_ap>0 else spk_arr[:, -pre_ap:]
        t_start = -t_pre
        t_stop = t_start + x_arr.shape[1]/s_f
    else:
        stop_ap = int(t_stop * s_f)
        x_arr = spk_arr[:, -pre_ap: stop_ap]
        t_start = -t_pre
        
        
    spk_train_list = [neo.SpikeTrain(times=np.where(x==1)[0]/s_f + t_start, 
                                     units='sec', 
                                     t_start=t_start, 
                                     t_stop=t_stop) for x in x_arr]
    
    return spk_train_list

def simple_spike_trains_from_spk_arr(spk_arr, s_f=30000):
    # get the time of the spikes
    # make a neo SpikeTrain for each cluster
    t_stop = spk_arr.shape[1]/s_f
    spk_train_list = [neo.SpikeTrain(times=np.where(x==1)[0]/s_f, units='sec', t_stop=t_stop) for x in spk_arr]
    
    return spk_train_list

def gpfa_for_bout(bout_df: pd.DataFrame, bout_dict: dict, 
bin_size_ms: int=15, x_dim: int=8, spk_train_kwargs: dict={}) -> tuple:
    # make the gpfa for all the bouts in a dataframe.
    # optional, use a unit mask

    # get the spike trains
    
    spk_train_kwargs['s_f'] = bout_dict['s_f_ap_0']
    logger.warning('Will use sampling frequency of ap_0 channel for spike trains')
    
    logger.info('Collecting spike trains')
    spk_train_list = bout_df['spk_arr'].apply(lambda x: spike_trains_from_spk_arr(x, **spk_train_kwargs))

    logger.info('Fitting gpfa model')
    bin_size = bin_size_ms * pq.ms
    gpfa_model = GPFA(bin_size=bin_size, x_dim=x_dim)
    #fit
    gpfa_model.fit(spk_train_list)
    
    logger.info('Projecting to latent dimensions')
    #project
    bout_fit_list = gpfa_model.transform(spk_train_list)
    
    logger.info('Done projecting')
    return gpfa_model, bout_fit_list

def spec_pca_for_bout(bout_df: pd.DataFrame, bout_dict: dict) -> pd.DataFrame:
    
    # collect all of the spectrogram traces
    all_x = np.hstack(bout_df['sxx'].apply(lambda x: x[2]))
    all_x[all_x==np.nan] = 1 ### better do a zscore

    # fit all
    pca = PCA(n_components=3, svd_solver='full')
    pca.fit(np.log(all_x.T))

    # project all
    bout_df['sxx_pc'] = bout_df['sxx'].apply(lambda x: pca.transform(np.log(x[2].T)))

    return bout_df
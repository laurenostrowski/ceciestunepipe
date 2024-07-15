import numpy as np
import pandas as pd
import warnings
import logging
from itertools import groupby
from operator import itemgetter
from matplotlib import pyplot as plt
from typing import Union

from ceciestunepipe.util.sound import spectral as sp

logger = logging.getLogger("ceciestunepipe.util.pressure")


def segment_pressure(x, threshold, s_f, min_syl_ms=100, max_syl_ms=1000, min_silence_ms=15, plot: bool=True):

    p_square = np.zeros_like(x)
    p_square[x>threshold] = 1
    
    min_syl_sample = int(min_syl_ms * s_f * 0.001)
    min_silence_sample = int(min_silence_ms * s_f * 0.001)
    #logger.info(min_syl_sample)
    
    # remove short silences if any
    silence = np.where(p_square<0.5)[0]
    sil_list = [(list(map(itemgetter(1), g))) for k, g in groupby(enumerate(silence), lambda ix: ix[0] - ix[1])]
    short_sil_list = [x for x in sil_list if len(x)<min_silence_sample]
    if len(short_sil_list) > 0:
        short_sil = np.concatenate(short_sil_list)
        p_square[short_sil] = 1
    
    
    # remove short syllables if any
    syllables = np.where(p_square>0.5)[0]
    syl_list = [(list(map(itemgetter(1), g))) for k, g in groupby(enumerate(syllables), lambda ix: ix[0] - ix[1])]
    short_syl_list = [x for x in syl_list if len(x)<min_syl_sample]
    if len(short_syl_list) > 0:
        short_syl = np.concatenate(short_syl_list)
        p_square[short_syl] = 0
    
    onset = np.where(np.diff(p_square)>0)[0]
    ofset = np.where(np.diff(p_square)<0)[0]
    # remove incomplente syllables
    if ofset[0] < onset[0]:
        ofset = ofset[1:]
    if onset[-1] > ofset[-1]:
        onset = onset[:-1]
    onoff = np.vstack([onset, ofset]).T
    
    # plot it
    if plot: 
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(x);
        ax.axhline(y=threshold, color='r')
        ax.plot(p_square, 'k-')
    #     p_square[short_sil] = 1
    #     p_square[short_syl] = 0
    #     ax.plot(p_square, 'k')
        ax.plot(onset, np.ones_like(onset), 'r*')
        ax.plot(ofset, np.zeros_like(ofset), 'g*')
    
    return onset, ofset, p_square, syl_list, sil_list

def segment_bout_pre(x: np.array, pre_kpa: np.array, p_thresh: float, bout_dict: dict, plot:bool=False) -> dict:
    s_f = bout_dict['s_f']
       
    try:
        onsets, ofsets, p_square, syl_list, sil_list = segment_pressure(pre_kpa, 
                                                                        p_thresh, 
                                                                        s_f, 
                                                                        min_syl_ms=100, 
                                                                        max_syl_ms=1000, 
                                                                        min_silence_ms=15, 
                                                                        plot=plot)
        
        x_segmented = {'onsets': onsets/s_f,
                       'offsets': ofsets/s_f}
        
    except:
        raise ValueError('Failed segmenting bout')
        
    # for consistency return in seconds
    return onsets/s_f, ofsets/s_f

def segment_series(ds: pd.Series, bout_dict: dict, p_thresh, plot=False) -> dict:
    #logger.info('segmenting bout {}'.format(ds['bout_idx']))
    try:
        on_sec,  off_sec = segment_bout_pre(ds['mic_arr'].flatten(), 
                                            ds['pre_kpa'].flatten(), 
                                            p_thresh, 
                                            bout_dict, 
                                            plot=plot)
        on = (on_sec * bout_dict['s_f']).astype(int)
        off = (off_sec * bout_dict['s_f']).astype(int)
    
    except ValueError:
        warnings.warn('Failed segmenting bout {}'.format(ds['start_sample']))
        on, off = (np.empty(0), np.empty(0)) # this will allow to keep the index  in the df for the bouts that could not be segmented
        # other option is on, off = (None, None)
    
#     on_ap = (on * bout_dict['s_f_ap_0']/bout_dict['s_f_nidq']).astype(int)
#     off_ap = (off * bout_dict['s_f_ap_0']/bout_dict['s_f_nidq']).astype(int)
    
#     on_gpfa = (on * ds['gpf_arr'].shape[-1]/ds['mic_arr'].flatten().size).astype(int)
        
    return on, off
import os
import pickle
import logging
import numpy as np
import pandas as pd
import warnings

from ceciestunepipe.file import bcistructure as et
from ceciestunepipe.util.spike import kilosort as ks
from ceciestunepipe.util import syncutil as su

logger = logging.getLogger('ceciestunepipe.util.sglxsync_debug')

def load_syn_dict(exp_struct: dict, stream:str, arrays_to_load=['evt_arr', 't_0']) -> dict:
    
    syn_dict_path = os.path.join(exp_struct['folders']['derived'],  
                                 '{}_sync_dict.pkl'.format(stream))
    logger.info('loading syn_dict from ' + syn_dict_path)
    with open(syn_dict_path, 'rb') as f:
        syn_dict = pickle.load(f)
     
    for arr in arrays_to_load:
        arr_path = syn_dict['{}_path'.format(arr)]
        syn_dict[arr] = np.load(arr_path, mmap_mode='r')
    return syn_dict


def get_syn_pattern(run_recs_dict, exp_struct, stream:str, force=False):
    logger.info('getting syn patterns for {}'.format(stream))
    syn_dict_path = os.path.join(exp_struct['folders']['derived'], '{}_sync_dict.pkl'.format(stream))
    
    # if Force=true of file not found, compute it from the recording_dict
    if not(os.path.exists(syn_dict_path) and (force is False)):
        logger.info('File {} not found or forced computation, getting the events'.format(syn_dict_path))
        syn_tuple = run_recs_dict[stream].get_effective_sf(force_ttl=True)
        syn_arr = np.vstack(list(syn_tuple[2][:]))
        
        t_0_path = os.path.join(exp_struct['folders']['derived'],  '{}_t0.npy'.format(stream))
        syn_npy_path = os.path.join(exp_struct['folders']['derived'],  '{}_sync_evt.npy'.format(stream))
                                 
        logger.info('saving events array to ' + syn_npy_path)
        np.save(syn_npy_path, syn_arr)
        
        logger.info('saving t_0 array to ' + t_0_path)                                 
        np.save(t_0_path, syn_tuple[1])
        
        syn_dict = {'path': syn_dict_path,
                    's_f': syn_tuple[0],
                   't_0_path': t_0_path,
                    'evt_arr_path': syn_npy_path
                   }
        
        # save without the array, and open the array as a memmap
        logger.info('saving sync dict to ' + syn_dict_path)
        with open(syn_dict_path, 'wb') as pf:
            pickle.dump(syn_dict, pf)
    
#     else:
#         syn_dict = load_syn_dict(exp_struct, stream, arrays_to_load=[])
#         syn_dict['path'] = syn_dict_path
#         # save without the array, and open the array as a memmap
#         logger.info('saving sync dict to ' + syn_dict_path)
#         with open(syn_dict_path, 'wb') as pf:
#             pickle.dump(syn_dict, pf)
            
    ## in any case, load the saved dict so everything comes from the memmaped arrays
    syn_dict = load_syn_dict(exp_struct, stream) 
        
    return syn_dict


def sync_all(all_syn_dict: dict, ref_stream: str, force=False) -> dict:
    logger.info('syncing all times to {}'.format(ref_stream))
    ref_syn_dict = all_syn_dict[ref_stream]
    for one_stream, one_syn_dict in all_syn_dict.items():
        if one_stream==ref_stream:
            continue
            
        logger.info(' sync {}...'.format(one_stream))
        
        t_0_folder = os.path.split(one_syn_dict['t_0_path'])[0]
        t_p_path = os.path.join(t_0_folder, '{}-tp.npy'.format(one_stream))
        
        if not(os.path.exists(t_p_path) and (force is False)):
            logger.info('  t_prime file {} not found or forced computation, getting the events'.format(t_p_path))
        
            # check if it had skipped beats
            skipped_beat = check_skipped(one_syn_dict)
            if skipped_beat:
                raise RuntimeError('Events array for {} had skipped heartbeats'.format(one_stream))

            t_prime = su.sync_to_pattern(one_syn_dict['evt_arr'], one_syn_dict['t_0'],
                                                     ref_syn_dict['evt_arr'], ref_syn_dict['t_0'])
            logger.info('    saving t_prime array to ' + t_p_path)                                 
            np.save(t_p_path, t_prime)
        
            #clar the memory, then load as memmap
            del t_prime
        
            one_syn_dict['t_p_path'] = t_p_path
            # save the dict with the path to the sync it
            logger.info('    saving synced dict to {}'.format(one_syn_dict['path']))
            with open(one_syn_dict['path'], 'wb') as fp:
                pickle.dump(one_syn_dict, fp)
            
        one_syn_dict['t_p'] = np.load(t_p_path, mmap_mode='r')
    logger.info('Done with sync_all')
    return


## Function added to adjust for mismatched stream lengths
## In the case that the neuropixel came unplugged during the recording,
##   ref_stream (ap_0) will be shorter than nidq and wav streams (short_stream = 'ap_0')
def sync_all_mismatched_streams(all_syn_dict: dict, ref_stream: str, short_stream: str, force=False) -> dict:
    logger.info('syncing all times to {}'.format(ref_stream))
    ref_syn_dict = all_syn_dict[ref_stream]
    
    new_len = np.shape(all_syn_dict[short_stream]['t_0'])[0]/all_syn_dict[short_stream]['s_f'] # length of short stream (s)
    ref_end = int(new_len * all_syn_dict[ref_stream]['s_f']) # where to truncate ref_stream (samples)
    len_evt_arr = np.shape(all_syn_dict[short_stream]['evt_arr'])[1] # number of events captured in short stream
    
    for one_stream, one_syn_dict in all_syn_dict.items():
        if one_stream==ref_stream:
            continue
            
        logger.info(' sync {}...'.format(one_stream))
        
        t_0_folder = os.path.split(one_syn_dict['t_0_path'])[0]
        t_p_path = os.path.join(t_0_folder, '{}-tp.npy'.format(one_stream))
        
        if not(os.path.exists(t_p_path) and (force is False)):
            logger.info('  t_prime file {} not found or forced computation, getting the events'.format(t_p_path))
            
            # Truncate one_stream to length of short_stream:
            one_end = int(new_len * all_syn_dict[one_stream]['s_f']) # where to truncate one_stream (samples)
            
            # Sync to ref stream
            t_prime = su.sync_to_pattern(one_syn_dict['evt_arr'][:,:len_evt_arr], one_syn_dict['t_0'][:one_end],
                                         ref_syn_dict['evt_arr'][:,:len_evt_arr], ref_syn_dict['t_0'][:ref_end])
            logger.info('    saving t_prime array to ' + t_p_path)
            np.save(t_p_path, t_prime)
        
            # clear the memory, then load as memmap
            del t_prime
        
            one_syn_dict['t_p_path'] = t_p_path
            # save the dict with the path to the sync it
            logger.info('    saving synced dict to {}'.format(one_syn_dict['path']))
            with open(one_syn_dict['path'], 'wb') as fp:
                pickle.dump(one_syn_dict, fp)
            
        one_syn_dict['t_p'] = np.load(t_p_path, mmap_mode='r')
    logger.info('Done with sync_all')
    return


def ignore_last_edge(all_syn_dict: dict, ref_stream:str):
    ## ignore the last edge in an array of ttls to be synced to a ref. stream
    ## It could happen that all other systems were shut down after the ref stream.
    ## This can be dangerous: first need to check 
    #   - that no beat was skipped,
    #   - that the sign of the first ttl match
    #   - that the sign of the remaining last ttl will match

    warnings.warn('About to discard heartbeat ttls beyond the last ttl in {}'.format(ref_stream))
    last_edge = all_syn_dict[ref_stream]['evt_arr'].size//2
    last_sign = all_syn_dict[ref_stream]['evt_arr'][1, -1]

    logger.info('Reference stream {} has {} edges, last is {}'.format(ref_stream, last_edge, last_sign))
    for stream, one_syn_dict in all_syn_dict.items():
        s_edge = one_syn_dict['evt_arr'].size//2
        s_sign = one_syn_dict['evt_arr'][1, -1]
        s_t = one_syn_dict['t_0'][one_syn_dict['evt_arr'][0, -1]]

        logger.info('stream {} had {} edges, last one {} at t={}'.format(stream, s_edge, s_sign, s_t))
        one_syn_dict['evt_arr'] = one_syn_dict['evt_arr'][:, :last_edge]
    
    return all_syn_dict

def check_skipped(one_syn_dict: dict, round_ms=50) -> int:
    # check if a beat was skipped
    skipped_beat = False

    evt_arr = one_syn_dict['evt_arr']
    evt_t = one_syn_dict['t_0'][evt_arr[0]]

    logger.info('Event array has {} events'.format(evt_arr.size//2))

    # get the unique periods, rounded at 50 ms
    evt_period_ms = np.unique(np.round((np.unique(np.diff(evt_t))*1000)/round_ms)*round_ms).astype(int)
    if evt_period_ms.size > 1:
        skipped_beat = True
        warnings.warn('More than 1 different periods detected: {}'.format(evt_period_ms))

    # check that the diff between every other edge is zero
    period_diff = np.hstack([np.diff(evt_arr[1][1:][::2]), np.diff(evt_arr[1][::2])])
    if not (all(period_diff==0)):
        skipped_beat=True
        warnings.warn('Difference between corresponding periodic edges is not zero: {}'.format(np.unique(period_diff)))
    
    return skipped_beat



def bout_dict_from_array(bout_arr: np.array, all_syn_dict: dict) -> dict:
    s_f_wav = all_syn_dict['wav']['s_f']
    
    start_ms = (bout_arr[:, 0]*1000).astype(np.int64)
    len_ms = (np.diff(bout_arr)*1000).astype(np.int64).flatten()
    
    
    bout_dict = {
            's_f': s_f_wav, # s_f used to get the spectrogram
            's_f_nidq': all_syn_dict['nidq']['s_f'],
            's_f_ap_0': all_syn_dict['ap_0']['s_f'],
           'start_ms': start_ms,
            'len_ms': len_ms,
           'start_sample_naive': ( start_ms * s_f_wav * 0.001).astype(np.int64),
           'start_sample_nidq': np.array([np.where(all_syn_dict['nidq']['t_0'] > start)[0][0] for start in start_ms*0.001]),
            'start_sample_wav': np.array([np.where(all_syn_dict['wav']['t_0'] > start)[0][0] for start in start_ms*0.001])
           }
    
    start_ms_ap_0 = all_syn_dict['wav']['t_p'][bout_dict['start_sample_wav']]*1000
    
    bout_dict['start_ms_ap_0'] = start_ms_ap_0
    bout_dict['start_sample_ap_0'] = np.array([np.where(all_syn_dict['ap_0']['t_0'] > start)[0][0] for start in start_ms_ap_0*0.001])
    bout_dict['start_sample_ap_0'] = (bout_dict['start_sample_ap_0']).astype(np.int64)
    return bout_dict

def bout_dict_from_pd(bout_pd: pd.DataFrame, all_syn_dict: dict, s_f_key: str='wav') -> dict:
    s_f = all_syn_dict[s_f_key]['s_f']

    start_ms = bout_pd['start_ms'].values
    len_ms = bout_pd['len_ms'].values

    bout_dict = {
        's_f': s_f, # s_f used to get the spectrogram
        's_f_nidq': all_syn_dict['nidq']['s_f'],
        's_f_ap_0': all_syn_dict['ap_0']['s_f'],
        'start_ms': start_ms,
        'len_ms': len_ms,
        'start_sample_naive': ( start_ms * s_f * 0.001).astype(np.int64),
        'start_sample_nidq': np.array([np.where(all_syn_dict['nidq']['t_0'] > start)[0][0] for start in start_ms*0.001]),
        'start_sample_wav': np.array([np.where(all_syn_dict['wav']['t_0'] > start)[0][0] for start in start_ms*0.001])
    }
    
    start_ms_ap_0 = all_syn_dict['wav']['t_p'][bout_dict['start_sample_wav']]*1000
    
    bout_dict['start_ms_ap_0'] = start_ms_ap_0
    bout_dict['start_sample_ap_0'] = np.array([np.where(all_syn_dict['ap_0']['t_0'] > start)[0][0] for start in start_ms_ap_0*0.001])
    bout_dict['start_sample_ap_0'] = (bout_dict['start_sample_ap_0']).astype(np.int64)
    bout_dict['end_sample_ap_0'] = bout_dict['start_sample_ap_0'] + (bout_dict['len_ms'] * bout_dict['s_f_ap_0'] * 0.001).astype(np.int64)
    
    ## update the bout pandas dataframe with the synced columns
    for k in ['start_ms_ap_0', 'start_sample_ap_0', 'len_ms', 'start_ms', 'start_sample_naive']:
        bout_pd[k] = bout_dict[k]

    return bout_dict, bout_pd

## Function added for the case that the neuropixel came unplugged during the recording
##   all bouts recorded after the neuropixel came unplugged will be discarded     -- Lo
def bout_dict_from_pd_mismatched_streams(bout_pd: pd.DataFrame, all_syn_dict: dict, s_f_key: str='wav') -> dict:
    s_f = all_syn_dict[s_f_key]['s_f']

    start_ms = bout_pd['start_ms'].values
    len_ms = bout_pd['len_ms'].values
    
    bout_dict = {
        's_f': s_f, # s_f used to get the spectrogram
        's_f_nidq': all_syn_dict['nidq']['s_f'],
        's_f_ap_0': all_syn_dict['ap_0']['s_f'],
        'start_ms': start_ms,
        'len_ms': len_ms,
        'start_sample_naive': ( start_ms * s_f * 0.001).astype(np.int64),
        'start_sample_nidq': np.array([np.where(all_syn_dict['nidq']['t_0'] > start)[0][0] for start in start_ms*0.001]),
        'start_sample_wav': np.array([np.where(all_syn_dict['wav']['t_0'] > start)[0][0] for start in start_ms*0.001])
    }
    
    # Edit to remove bout starts > length of ap_0 recording
    keep = bout_dict['start_sample_wav'] <= len(all_syn_dict['wav']['t_p'])
    bout_dict['start_ms'] = bout_dict['start_ms'][keep]
    bout_dict['len_ms'] = bout_dict['len_ms'][keep]
    bout_dict['start_sample_naive'] = bout_dict['start_sample_naive'][keep]
    bout_dict['start_sample_nidq'] = bout_dict['start_sample_nidq'][keep]
    bout_dict['start_sample_wav'] = bout_dict['start_sample_wav'][keep]
    
    if bout_dict['start_sample_wav'].size > 0:
        start_ms_ap_0 = all_syn_dict['wav']['t_p'][bout_dict['start_sample_wav']]*1000
    
        bout_dict['start_ms_ap_0'] = start_ms_ap_0
        bout_dict['start_sample_ap_0'] = np.array([np.where(all_syn_dict['ap_0']['t_0'] > start)[0][0] for start in start_ms_ap_0*0.001])
        bout_dict['start_sample_ap_0'] = (bout_dict['start_sample_ap_0']).astype(np.int64)
        bout_dict['end_sample_ap_0'] = bout_dict['start_sample_ap_0'] + (bout_dict['len_ms'] * bout_dict['s_f_ap_0'] * 0.001).astype(np.int64)
    
    else:
        bout_dict['start_ms_ap_0'] = []
        bout_dict['start_sample_ap_0'] = []
        bout_dict['start_sample_ap_0'] = []
        bout_dict['end_sample_ap_0'] = []
    
    ## update the bout pandas dataframe with the synced columns
    bout_pd = bout_pd.head(keep.sum()) # trim bout_pd to bouts within ap_0 recording
    for k in ['start_ms_ap_0', 'start_sample_ap_0', 'len_ms', 'start_ms', 'start_sample_naive']:
        pd.options.mode.chained_assignment = None
        bout_pd[k] = bout_dict[k]

    return bout_dict, bout_pd


def trial_syn_from_pd(bout_pd: pd.DataFrame, all_syn_dict: dict, s_f_key: str='nidq') -> dict:
    
    s_f_nidq = all_syn_dict['nidq']['s_f']
    
    start_sample = bout_pd['start'].values
    len_sample = (bout_pd['end'] - bout_pd['start']).values
    start_s = all_syn_dict['nidq']['t_0'][start_sample]
    
    bout_pd['start_ms'] = start_s * 1000
    bout_pd['len_ms'] = len_sample / s_f_nidq * 1000
    
    # overwrite the start sample with the one that I got from the synpd
    bout_dict, bout_pd = bout_dict_from_pd(bout_pd, all_syn_dict, s_f_key=s_f_key)
    
    bout_pd['start_sample_' + s_f_key] = start_sample
    bout_dict['start_sample_' + s_f_key] = start_sample
    # rename start, end to start_sample, end sample, for compatibility with bouts pipeline.
    bout_pd.rename(columns={'start': 'start_sample', 'end': 'end_sample'}, inplace=True)
    
    bout_dict.update({
            'tag_freq': bout_pd['tag_freq'].values,
            'stim_name': bout_pd['stim_name'].values
           })
    return bout_dict, bout_pd


def sync_trial_pd(trial_pd: pd.DataFrame, all_syn_dict: dict, keep_keys: list=None, s_f_key: str='nidq'):
    ### sync all the trials into a dictionary
    ### use the dictionary to complete a synced_trial_pd
    
    trial_dict, trial_syn_pd = trial_syn_from_pd(trial_pd, all_syn_dict, s_f_key=s_f_key)
    
    if keep_keys is None:
        keep_keys = ['start_ms', 'len_ms', 'tag_freq','start_sample_nidq', 'start_sample_naive', 
        'start_sample_ap_0', 'start_ms_ap_0']
    
    dict_to_pd = {k: v for (k, v) in trial_dict.items() if k in keep_keys}
    
    synced_trial_pd = pd.DataFrame.from_dict(dict_to_pd)
    synced_trial_pd = trial_pd.merge(synced_trial_pd, left_on='start', right_on='start_sample_' + s_f_key)
    
    return trial_dict, synced_trial_pd

def collect_bout(bout_dict, bout_idx, run_recordings, t_pre, t_post, spk_df, clu_list, mic_stream, all_syn_dict):
    ## add the length of the bout (in seconds) to the end of the segment
    t_post += int(bout_dict['len_ms'][bout_idx] * 0.001)
       
    start_ap = bout_dict['start_sample_ap_0'][bout_idx] + int(all_syn_dict['ap_0']['s_f']* t_pre)
    end_ap = bout_dict['start_sample_ap_0'][bout_idx] + int(all_syn_dict['ap_0']['s_f'] * t_post)

    start_wav = bout_dict['start_sample_wav'][bout_idx] + int(all_syn_dict['wav']['s_f'] * t_pre)
    end_wav = bout_dict['start_sample_wav'][bout_idx] + int(all_syn_dict['wav']['s_f'] * t_post)


    # get the streams/spike array
    spk_arr = ks.get_window_spikes(spk_df, clu_list, int(start_ap), int(end_ap))
    mic_arr = mic_stream.flatten()[start_wav: end_wav]
    
    return spk_arr, mic_arr

def collect_all_bouts(bout_dict:dict, clu_list: list, run_recordings:dict, spk_df: pd.DataFrame, 
t_pre: float=-5, 
t_post:float=5) -> pd.DataFrame:
    
    spk_arr_list = []
    mic_arr_list = []
    clu_id_arr_list = []

    t_pre = -5
    t_post = 5

    ## get the bouts arrays
    for bout_idx, start in enumerate(bout_dict['start_ms']):
        spk_arr, mic_arr = collect_bout(bout_dict, bout_idx, run_recordings, t_pre, t_post, spk_df, clu_list, mic_stream)
        spk_arr_list.append(spk_arr.astype(np.short))
        mic_arr_list.append(mic_arr.astype(np.int16))
        clu_id_arr_list.append(np.array(clu_list))
        

    ## make into a pandas dataframe
    bout_dict['t_pre_ms'] = t_pre * 1000
    bout_dict['t_post_ms'] = t_post * 1000

    bout_dict['spk_arr'] = spk_arr_list
    bout_dict['mic_arr'] = mic_arr_list
    bout_dict['clu_id_arr'] = clu_id_arr_list
    keys_to_df = ['start_sample_nidq', 'start_sample_ap_0', 'len_ms', 'spk_arr', 'mic_arr', 'clu_id_arr']

    bout_dict_df = {k: bout_dict[k] for k in keys_to_df}
    bout_df = pd.DataFrame.from_dict(bout_dict_df)

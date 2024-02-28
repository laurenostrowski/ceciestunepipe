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
import json

from scipy.io import wavfile

from ceciestunepipe.util import syncutil as su

logger = logging.getLogger('ceciestunepipe.util.wavutil')

def save_npy_wav(npy_path: str, s_f: float, stream: np.array):
    ## make a dict with the metadata of the npy file
    meta_dict = {'s_f': s_f,
                 'dtype': stream.dtype,
                 'shape':stream.shape}
    
    json_meta_dict = {'s_f': s_f,
                 'dtype': str(stream.dtype),
                 'shape':stream.shape}
    
    logger.info('saving {}-shaped array as npy in {}'.format(stream.shape, npy_path))
    np.save(npy_path, stream)
    
    json_path = npy_path.split('.')[0] + '-npy_meta.json'
    logger.info('Saving meta of npy array to {}'.format(json_path))
    with open(json_path, 'w') as fp:
        json.dump(json_meta_dict, fp)
    
    pkl_path = npy_path.split('.')[0] + '-npy_meta.pickle'
    logger.info('Saving meta of npy array to {}'.format(pkl_path))
    with open(pkl_path, 'wb') as fp:
        pickle.dump(meta_dict, fp)
    return meta_dict

def load_npy_wav(npy_path: str, mmap=True):
    # load the array 
    # load the metadata with the shape, dtype of the array
    # shape the array
    logger.info('loading npy array from {}'.format(npy_path))

    pkl_path = npy_path.split('.')[0] + '-npy_meta.pickle'
    with open(pkl_path, 'rb') as fp:
        meta_dict = pickle.load(fp)
    
    mmap_mode = 'r' if mmap else None
    x = np.load(npy_path, mmap_mode=mmap_mode).astype(meta_dict['dtype']).reshape(meta_dict['shape'])
    
    return meta_dict['s_f'], x

def save_wav(stream: np.array, s_f: float, wav_path: str, skip_wav: bool=False):
    wav_s_f = int(round(s_f/1000))*1000
    logger.info('sampling rate {}'.format(wav_s_f))
    
    # write to wav
    logger.info('saving {}-shaped array as wav in {}'.format(stream.shape, wav_path))
    
    os.makedirs(os.path.split(wav_path)[0], exist_ok=True, mode=0o777)
    if not skip_wav:
        wavfile.write(wav_path, wav_s_f, stream.T)
    else:
        logger.info('Not saving the file as wav, going straight to numpy + dict')
    ## also save as numpy
    npy_path = wav_path.split('.')[0] + '.npy'
    logger.info('saving {}-shaped array as npy in {}'.format(stream.shape, npy_path))
    save_npy_wav(npy_path, wav_s_f, stream.T)
    return wav_s_f

def read_wav_chan(wav_path: str, chan_id: int=0, skip_wav=False) -> tuple:
    if skip_wav:
        npy_path = wav_path.split('.')[0] + '.npy'
        logger.info('skipping wav, loading npy instead from {}'.format(npy_path))
        s_f, x = load_npy_wav(npy_path, mmap=True)
    else:
        try:
            s_f, x = wavfile.read(wav_path, mmap=True)
        except FileNotFoundError:
            logger.info('Did not find wav file {}. Will try loading npy + dict'.format(wav_path))
            return read_wav_chan(wav_path, chan_id, skip_wav=True)
    if x.ndim==1:
        if chan_id > 0:
            raise ValueError('Wave file has only one channel, asking for channel {}'.format(chan_id))
        x = x.reshape(-1, 1)
    
    return s_f, x[:, chan_id]

def wav_to_syn(file_path:str, chan=0, skip_wav=False):
    # read the channel from the wave
    # turn the square into a 'digital square' (e.g set a threshold)
    s_f, x = read_wav_chan(file_path, chan_id=chan, skip_wav=skip_wav)
    ttl_thresh = su.quick_ttl_threshold(x)
    
    # make a short 'digital' x
    x_dig = np.zeros_like(x, dtype=np.int8)
    x_dig[x > ttl_thresh] = 1
    
    # get the edges of it
    ttl_events = su.square_to_edges(x_dig)
    ttl_arr = np.vstack(list(ttl_events[:]))
    
    return s_f, x_dig, ttl_arr
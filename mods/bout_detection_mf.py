import librosa
from scipy.io import loadmat
from scipy.signal import filtfilt, lfilter
import numpy as np
import pandas as pd
import pickle
import logging
import sys
from tqdm.auto import tqdm


logger = logging.getLogger('ceciestunepipe.mods.bout_detection_mf')


def load_mult_mics(file_path: str) -> tuple:
    # will return y = (stream_length, n_mics)
    
    npy_path = file_path.split('.')[0] + '.npy'
    pkl_path = npy_path.split('.')[0] + '-npy_meta.pickle'
    with open(pkl_path, 'rb') as fp:
        meta_dict = pickle.load(fp)
    x = np.load(npy_path, mmap_mode='r').astype(meta_dict['dtype']).reshape(meta_dict['shape'])
    s_f = meta_dict['s_f']
    
    if x.dtype == 'int32':
        y = (x >> 16).astype(np.int16)
    elif x.dtype == 'int16':
        y = x
    else:
        raise NotImplementedError('wav file is neither int16 nor int32')
    
    return s_f, y


def load_filter_coefficients_matlab(filter_file_path):
    """"
    Load Butterworth filter coefficients from a file
    """
    coefficients = loadmat(filter_file_path)
    a = coefficients['a'][0]
    b = coefficients['b'][0]
    return b, a  # The output is a double list after loading .mat file

def noncausal_filter(signal, b, a=1):
    """"
    Perform non-causal filter given filter coefficients
    """
    y = filtfilt(b, a, signal)
    return y

def calculate_signal_rms(signal):
    """"
     Returns the root mean square {sqrt(mean(samples.^2))} of a 1D vector.
    """
    return np.sqrt(np.mean(np.square(signal)))

def find_start_end_idxs_POIs(binary_signal, samples_between_poi, min_samples_poi=1):
    """"
    Returns a list of tuples (start_idx, end_idx) for each period of interest found in the audio file.
    
     Input: Binary vector where ones indicate samples that are above a specified audio threshold, zeros indicate samples below the threshold.
     samples_between_poi = Number of samples needed to consider two POIs independent.
     min_samples_poi = Minimum number of samples that a POI must have to not be discarded.

     Output: list of [start_idx, end_idx] for each POI found.
    """
    start_end_idxs = []
    start_idx = None 
    end_idx = None 
    zero_counter = 0

    for i in range(len(binary_signal)):

        if binary_signal[i] == 1:
            if start_idx == None:
                start_idx = i

            elif zero_counter != 0 or end_idx is not None:
                zero_counter = 0
                end_idx = None

        elif binary_signal[i] == 0:

            if start_idx != None:
                if zero_counter == 0: 
                    end_idx = i
                    zero_counter += 1

                elif zero_counter < samples_between_poi:
                    zero_counter += 1

                elif zero_counter >= samples_between_poi:
                    if end_idx - start_idx > min_samples_poi:  
                        start_end_idxs.append([start_idx, end_idx])
                    start_idx = None
                    end_idx = None
                    zero_counter = 0

        # if we are in a POI and the file ends
        if i == len(binary_signal)-1 and start_idx != None:
            start_end_idxs.append([start_idx, i])
            
    return start_end_idxs

def preemphasis(x, hparams):
    return lfilter([1, -hparams['preemphasis']], [1], x)

def stft(y, hparams):
    n_fft = (hparams['num_freq'] - 1) * 2
    hop_length = int(hparams['frame_shift_ms'] / 1000 * hparams['sample_rate'])
    win_length = int(hparams['frame_length_ms'] / 1000 * hparams['sample_rate'])
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def normalize(S, hparams):
    return np.clip((S - hparams['min_level_db']) / -hparams['min_level_db'], 0, 1)

def rosa_spectrogram(y, hparams):
    D = stft(preemphasis(y, hparams), hparams)
    S = amp_to_db(np.abs(D)) - hparams['ref_level_db']
    return normalize(S, hparams)

def gimmepower(x, hparams):
    s = rosa_spectrogram(x.flatten(), hparams)
    f = np.arange(hparams['num_freq']) /         hparams['num_freq']*0.5*hparams['sample_rate']
    s = s[(f > hparams['fmin']) & (f < hparams['fmax']), :]
    f = f[(f > hparams['fmin']) & (f < hparams['fmax'])]
    p = s.sum(axis=0)
    return p, f, s


def get_bouts_in_long_file(file_path, hparams, loaded_p=None, chunk_size=50000000):
    
    logger.info('Getting bouts for long file {}'.format(file_path))
    sys.stdout.flush()
    
    # bout detection parameters
    th = 8  # amplitude detection threshold: # of standard deviations above RMS
    time_between_poi = 2  # minimum silence (s) between POIs
    min_poi_time = 0.5  # minimum duration (s) of POI
    
    # filter coefficients
    b, a = load_filter_coefficients_matlab('/mnt/cube/lo/envs/ceciestunepipe/filters/butter_bp_250hz-8000hz_order4_sr48000.mat')

    # get the bouts
    sr, wav_i = load_mult_mics(file_path)
    hparams['sample_rate'] = sr
    
    if wav_i.shape[1] == 1:
        female_present = False
        logger.info('loading mic stream')
    elif wav_i.shape[1] == 2:
        # separate male and female streams
        female_present = True
        logger.info('loading male and female mic streams -- will perform bout detection on male mic stream only')
    else:
        raise ValueError(f"Expected 1-2 mic streams but received {wav_i.shape[1]}.")
    
    wav_m = wav_i[:, 0] # male stream
    if female_present: wav_f = wav_i[:, 1] # female stream
    
    n_chunks = int(np.ceil(wav_m.shape[0]/chunk_size))
    wav_chunks = np.array_split(wav_m, n_chunks)
    if female_present:
        wav_chunks_f = np.array_split(wav_f, n_chunks)
    logger.info('splitting file into {} chunks'.format(n_chunks))

    chunk_offset = 0
    bouts_pd_list = []
    for i_chunk, wav_chunk in tqdm(enumerate(wav_chunks), total=n_chunks):
        if female_present: wav_chunk_f = wav_chunks_f[i_chunk] # matching female stream
        
        # filter audio
        filt_audio_signal = noncausal_filter(wav_chunk, b, a=a)
        # rectify audio signal
        rf_filt_audio_signal = np.absolute(filt_audio_signal)
        # calculate RMS of audio signal
        rms = calculate_signal_rms(rf_filt_audio_signal)
        # create binary vector of indices where the audio crosses the specified threshold
        idx_above_th = np.argwhere(rf_filt_audio_signal > th*rms)
        binary_signal = np.zeros(len(rf_filt_audio_signal))
        binary_signal[idx_above_th] = 1
        # retrieve start / end sample index for each POI found.
        start_end_idxs = find_start_end_idxs_POIs(binary_signal, time_between_poi*sr, min_samples_poi=min_poi_time*sr)
        
        # loop through POIs detected
        if not start_end_idxs: # list is empty
            bout_pd = pd.DataFrame()
        else:
            data = []
            for poi, poi_idxs in enumerate(start_end_idxs):
                try:
                    # include extra second on either side of detection if possible
                    if poi_idxs[0]-sr >= 0:
                        start_idx = poi_idxs[0]-sr
                    else:
                        start_idx = poi_idxs[0]
                    if poi_idxs[1]+sr <= len(wav_chunk):
                        end_idx = poi_idxs[1]+sr
                    else:
                        end_idx = poi_idxs[1]
                    
                    # extract signal
                    signal = wav_chunk[start_idx:end_idx]
                    if female_present: signal_f = wav_chunk_f[start_idx:end_idx]

                    # calculate spectrogram
                    _, _, spec = gimmepower(signal, hparams)
                    
                    # adjust sample times for chunk offset
                    start_idx += chunk_offset
                    end_idx += chunk_offset
                    
                    # 
                    start_ms = int(np.floor(start_idx/sr*1000))
                    end_ms = int(np.ceil(end_idx/sr*1000))
                    len_ms = end_ms-start_ms
                    
                    # store values
                    if female_present:
                        data.append({
                            'file': file_path,
                            'start_sample': start_idx,
                            'end_sample': end_idx,
                            'start_ms': start_ms,
                            'end_ms': end_ms,
                            'len_ms': len_ms,
                            'waveform': signal,
                            'fem_waveform': signal_f,
                            'spectrogram': spec,
                            'sample_rate': sr
                        })
                        
                    else:
                        data.append({
                            'file': file_path,
                            'start_sample': start_idx,
                            'end_sample': end_idx,
                            'start_ms': start_ms,
                            'end_ms': end_ms,
                            'len_ms': len_ms,
                            'waveform': signal,
                            'spectrogram': spec,
                            'sample_rate': sr
                        })

                except:
                    continue

            # make one bouts pd dataframe for the chunk
            bout_pd = pd.DataFrame(data)
        
        # append bout_pd to list and update start sample
        bouts_pd_list.append(bout_pd)
        chunk_offset += wav_chunk.size

    all_bout_pd = pd.concat(bouts_pd_list)
    all_bout_pd.reset_index(inplace=True, drop=True)
    
    return all_bout_pd

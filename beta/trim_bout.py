## Called by 2-curate_bouts in the chronic_ephys processing pipeline

import peakutils
import numpy as np
import pandas as pd
import pickle
import logging
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from ceciestunepipe.util.sound import boutsearch as bs
from ceciestunepipe.util.sound import spectral as sp
import noisereduce as nr
import matplotlib.pyplot as plt
from ipywidgets import widgets
from traitlets import CInt, link
from IPython.display import display, clear_output


logger = logging.getLogger('ceciestunepipe.lo.trim_bout')


def get_power_in_bouts(bout_pd: pd.DataFrame, hparams):
    """
     Generate 'p_step', 'peak_p', 'waveform', 'valid_waveform', 'valid', and 'spectrogram'

    Args:
        bout_pd (pandas.DataFrame): bouts dataframe
        hparams (dict): bout hyperparameters

    Returns:
        pandas.DataFrame: updated bouts dataframe
    """
    # Extract bouts
    start_samples = bout_pd['start_sample'].values
    end_samples = bout_pd['end_sample'].values
    wav_data = [hparams['read_wav_fun'](file)[1] for file in bout_pd['file']]
    bouts = [wav_i[start:end] for (start, end, wav_i) in zip(start_samples, end_samples, wav_data)]
    
    # Calculate power
    logger.debug('Computing power')
    power_values = [bs.gimmepower(x, hparams)[0] for x in bouts]
    bout_pd['p_step'] = power_values
    bout_pd['peak_p'] = bout_pd['p_step'].apply(np.max)
    
    # Extract waveform for each bout with a delta cushion on either side
    try:
        delta = int(hparams['waveform_edges'] *
                    hparams['sample_rate'] * 0.001)
    except KeyError:
        delta = 0
    bout_pd['waveform'] = [wav_i[start-delta:end+delta] for (start, end, wav_i) in zip(start_samples, end_samples, wav_data)]
    
    # Drop any empty waveforms (shouldn't happen, but Zeke had as potential edge case)
    bout_pd = bs.cleanup(bout_pd)
    
    # Calculate spectrogram for each bout
    bout_pd['spectrogram'] = bout_pd['waveform'].apply(lambda x: bs.gimmepower(x, hparams)[2])
    
    return bout_pd


def process_peaks(bout_pd: pd.DataFrame, all_syl, hparams):
    """
     Generate 'syl_in', 'n_syl', 'peaks_p', 'n_peaks', and 'l_p_ratio'

    Args:
        bout_pd (pandas.DataFrame): bouts dataframe
        all_syl (numpy.ndarray): 2D numpy array with onset and offset per identified syllable
        hparams (dict): bout hyperparameters

    Returns:
        pandas.DataFrame: updated bouts dataframe
    """
    # Create syllable dataframe
    syl_pd = pd.DataFrame(all_syl, columns=['start_ms', 'end_ms']) # all_syl coming from original bout_pd
    
    # Extract start_ms and end_ms for every syllable (ID'd by waveform thresholding) that appears in every bout
    bout_pd['syl_in'] = bout_pd.apply(lambda r:
                                      syl_pd[(syl_pd['start_ms'] >= r['start_ms']) &
                                             (syl_pd['start_ms'] <= r['end_ms'])].values,
                                      axis=1)
    # Calculate number of syllables
    bout_pd['n_syl'] = bout_pd['syl_in'].apply(len)
    
    # Get all the peaks larger than the threshold (peak_thresh_rms * rms)
    step_ms = hparams['frame_shift_ms']
    pk_dist = hparams['min_segment']
    bout_pd['peaks_p'] = bout_pd.apply(lambda r: peakutils.indexes(r['p_step'],
                                                                   thres=hparams['peak_thresh_rms'] *
                                                                   r['rms_p'] /
                                                                   r['p_step'].max(),
                                                                   min_dist=pk_dist//step_ms),
                                       axis=1)
    
    # Calculate number of peaks
    bout_pd['n_peaks'] = bout_pd['peaks_p'].apply(len)
    
    # Calculate length-to-peak ratio
    bout_pd['l_p_ratio'] = bout_pd.apply(
        lambda r: np.nan if r['n_peaks'] == 0 else r['len_ms'] / (r['n_peaks']), axis=1)
       
    return bout_pd


def bout_dict_from_pd(bout_pd: pd.DataFrame, syn_dict_path: str) -> dict:
    """
     Generate 'syl_in', 'n_syl', 'peaks_p', 'n_peaks', and 'l_p_ratio'

    Args:
        bout_pd (pandas.DataFrame): bouts dataframe
        syn_dict_path (str): path to sync dicts created during preprocessing

    Returns:
        pandas.DataFrame: updated bouts dataframe
        dict: bouts dictionary
    """
    # Rebuild all_syn_dict --> not saved in preprocessing
    suffix = '_sync_dict.pkl'
    syn_dict_wav_path = os.path.join(syn_dict_path, 'wav' + suffix)
    syn_dict_nidq_path = os.path.join(syn_dict_path, 'nidq' + suffix)
    syn_dict_ap_0_path = os.path.join(syn_dict_path, 'ap_0' + suffix)
    syn_dict_lf_0_path = os.path.join(syn_dict_path, 'lf_0' + suffix)
    
    with open(syn_dict_wav_path, 'rb') as handle:
        syn_dict_wav = pickle.load(handle)
    try: 
        syn_dict_wav['t_p']
    except KeyError:
        # sometimes t_p isn't saved in wav dict, but t_p path is
        syn_dict_wav['t_p'] = np.load(syn_dict_wav['t_p_path'])
    
    with open(syn_dict_nidq_path, 'rb') as handle:
        syn_dict_nidq = pickle.load(handle)
        
    with open(syn_dict_ap_0_path, 'rb') as handle:
        syn_dict_ap_0 = pickle.load(handle)
    try:
        syn_dict_ap_0['t_0']
    except KeyError:
        # sometimes t_0 isn't saved in ap_0 dict, but t_0 path is
        syn_dict_ap_0['t_0'] = np.load(syn_dict_ap_0['t_0_path'])
        
    with open(syn_dict_lf_0_path, 'rb') as handle:
        syn_dict_lf_0 = pickle.load(handle)
    
    all_syn_dict = {'wav': syn_dict_wav,
                   'nidq': syn_dict_nidq,
                   'ap_0': syn_dict_ap_0,
                   'lf_0': syn_dict_lf_0}
    
    # Build bout_dict using trimmed bout start_ms and end_ms values
    s_f = all_syn_dict['wav']['s_f']
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
    
    # Update trimmed bout_df with the synced columns
    for k in ['start_ms_ap_0', 'start_sample_ap_0', 'len_ms', 'start_ms', 'start_sample_naive']:
        bout_pd[k] = bout_dict[k]

    return bout_dict, bout_pd


def handle_trim_bouts(original_bout_df, syn_dict_path, trimmed_start_ms, trimmed_end_ms, hparams):
    """
     Trim bouts and repopulate all entries in bouts dataframe

    Args:
        original_bout_df (pandas.DataFrame): bouts dataframe
        syn_dict_path (str): path to sync dicts created during preprocessing
        trimed_start_ms (numpy.dnarray): bout onsets in ms
        trimed_end_ms (numpy.dnarray): bout offsets in ms
        hparams (dict): bout hyperparameters

    Returns:
        pandas.DataFrame: updated bouts dataframe
        dict: bouts dictionary
    """
    ### Might need to chunk to handle very large bout_dfs -- will prob run out of memory
    
    # Create trimmed bout_df with values carried over from original bout_df
    trimmed_bout_df = original_bout_df[['file', 'bout_check', 'confusing', 'is_call', 'rms_p']].copy()
    
    # Collect trimmed start_ms, len_ms, and end_ms
    trimmed_bout_df['start_ms'] = original_bout_df['start_ms'] + trimmed_start_ms
    trimmed_bout_df['len_ms'] = trimmed_end_ms - trimmed_start_ms
    trimmed_bout_df['end_ms'] = trimmed_bout_df['start_ms'] + trimmed_bout_df['len_ms']
    
    # Recalculate 'start_sample' and 'end_sample'
    with open(os.path.join(syn_dict_path, 'wav_sync_dict.pkl'), 'rb') as handle:
        syn_dict_wav = pickle.load(handle)
    trimmed_bout_df['start_sample'] = trimmed_bout_df['start_ms'] * (syn_dict_wav['s_f']//1000)
    trimmed_bout_df['end_sample'] = trimmed_bout_df['end_ms'] * (syn_dict_wav['s_f']//1000)
    
    # Generate 'p_step', 'peak_p', 'waveform', 'valid_waveform', 'valid', and 'spectrogram'
    trimmed_bout_df = get_power_in_bouts(trimmed_bout_df, hparams)
    
    # Recalculate 'syl_in', 'n_syl', 'peaks_p', 'n_peaks', and 'l_p_ratio' 
    all_syl = np.concatenate(original_bout_df['syl_in'])
    trimmed_bout_df = process_peaks(trimmed_bout_df, all_syl, hparams)
    
    # Make bout_dict and recalculate 'start_ms_ap_0', 'start_sample_ap_0', 'start_sample_naive'
    bout_dict, trimmed_bout_df = bout_dict_from_pd(trimmed_bout_df, syn_dict_path)
    
    # Reorder dataframe columns to match original order
    trimmed_bout_df = trimmed_bout_df[original_bout_df.columns]
    
    return trimmed_bout_df, bout_dict



### Graphing functions

def butter_filt(fs, lowcut = [], highcut = [], btype='band', order=5):
    """
    Apply a Butterworth filter to the data

    Args:
        data (numpy.ndarray): Input data
        lowcut (float): Low-frequency cutoff (optional)
        highcut (float): High-frequency cutoff (optional)
        btype (str): Filter type ('band', 'low', or 'high')

    Returns:
        numpy.ndarray: Filtered data
    """
    assert btype in ['band', 'low', 'high'] , "Filter type must be 'low', 'high', or 'band'"
    nyq = 0.5 * fs
    if lowcut: low = lowcut / nyq
    if highcut: high = highcut / nyq

    if btype == 'band': b, a = butter(order, [low, high], btype)
    elif btype == 'low': b, a = butter(order, low, btype)
    elif btype == 'high': b, a = butter(order, high, btype)
    return b, a

def noncausal_filter(data, b, a=1):
    y = filtfilt(b, a, data)
    return y


    
### WIDGET

# Create a counter object
class Counter(widgets.DOMWidget):
    value = CInt(0)
    value.tag(sync=True)


# Master class
class TrimBout():
    def __init__(self, hparams, bouts_pd):
        self.bouts_pd = bouts_pd
        self.init_crop_vals = np.column_stack((np.zeros(len(bouts_pd)), # initialize to full bout
                                               bouts_pd['len_ms'].values/1000))
        self.crop_min = self.init_crop_vals[:,0]
        self.crop_max = self.init_crop_vals[:,1]
        self.bout_series = None
        self.bout_counter = None
        self.bout_id = None
        self.buttons = {}
        self.m_pick = None   # bout candidate index slider
        self.crop_start = None
        self.crop_end = None
        self.x = None   # mic array
        self.fs = hparams['sample_rate']
        self.delta = hparams.get('waveform_edges', 0)*self.fs//1000
        self.init_fig()
        self.init_widget()
        self.show()
    
    def init_fig(self):
        self.song_fig, self.ax = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [1,3]}, figsize=(10,4), constrained_layout=True)
    
    def init_widget(self):
        self.bout_counter = Counter()
        # buttons
        self.buttons['Next'] = widgets.Button(description="Next", button_style='info',icon='plus')   
        self.buttons['Prev'] = widgets.Button(description="Prev", button_style='warning',icon='minus')
        [b.on_click(self.button_click) for b in self.buttons.values()]
        button_box = widgets.HBox([self.buttons['Prev'], self.buttons['Next']])
        # index slider
        self.m_pick = widgets.IntSlider(value=0, min=0, max=self.bouts_pd.index.size-1, step=1, description="Bout idx:")
        link((self.m_pick, 'value'), (self.bout_counter, 'value'))
        self.m_pick.observe(self.slider_change, names='value')
        # crop text entry box
        self.crop_start = widgets.BoundedFloatText(
            value=self.init_crop_vals[self.bout_counter.value,0],
            min=self.init_crop_vals[self.bout_counter.value,0],
            max=self.init_crop_vals[self.bout_counter.value,1],
            description = 'Crop:',
            disabled=False)
        self.crop_end = widgets.BoundedFloatText(
            value=self.init_crop_vals[self.bout_counter.value,1],
            min=self.init_crop_vals[self.bout_counter.value,0],
            max=self.init_crop_vals[self.bout_counter.value,1],
            disabled=False)
        text_box = widgets.HBox([self.crop_start, self.crop_end])
        # display
        self.update_bout()
        display(widgets.VBox([button_box, self.m_pick, text_box]))
    
    def button_click(self, button):
        self.bout_id = self.bout_counter.value
        curr_bout = self.bout_counter
        if button.description == 'Next':
            curr_bout.value += 1
        elif button.description == 'Prev':
            curr_bout.value -= 1
        if curr_bout.value > self.m_pick.max:
            curr_bout.value = 0 
        if curr_bout.value < self.m_pick.min:
            curr_bout.value = self.m_pick.max
    
    def slider_change(self, change):
        self.update_bout()
        self.show()
    
    def update_bout(self):
        # store crop vals
        if self.bout_id is not None:
            self.crop_min[self.bout_id] = self.crop_start.value
            self.crop_max[self.bout_id] = self.crop_end.value
        # update bout id
        self.bout_id = self.bout_counter.value
        self.bout_series = self.bouts_pd.iloc[self.bout_id]
        self.x = self.bout_series['waveform'][self.delta:-self.delta]
        # update crop_start
        self.crop_start.min = self.init_crop_vals[self.bout_id,0]
        self.crop_start.max = self.init_crop_vals[self.bout_id,1]
        self.crop_start.value = self.crop_min[self.bout_id]
        # update crop_end
        self.crop_end.min = self.init_crop_vals[self.bout_id,0]
        self.crop_end.max = self.init_crop_vals[self.bout_id,1]
        self.crop_end.value = self.crop_max[self.bout_id]
        
    def show(self):
        [ax.cla() for ax in self.ax]
        # Lowpass filter
        b, a = butter_filt(self.fs, highcut = 1600, btype='high')
        mic_arr_hp = noncausal_filter(self.x, b, a)
        # Tim Sainburg's noise reduce algorithm
        mic_arr_nr = nr.reduce_noise(self.x, self.fs, n_std_thresh_stationary=0.5)
        # Calculate spectrogram
        f, t, sxx = sp.ms_spectrogram(mic_arr_hp.flatten(), self.fs)
        # Graph sonogram
        self.ax[0].plot(mic_arr_nr.flatten(), 'black')
        self.ax[0].set_xlim([0, len(mic_arr_nr.flatten())])
        self.ax[0].set_axis_off()
        # Graph spectrogram
        self.ax[1].pcolormesh(t, f, np.log(sxx), cmap='inferno')
        self.ax[1].set_xlabel('time (s)', fontsize=16)
        self.ax[1].set_xlim([t[0], t[-1]])
        self.ax[1].tick_params(axis='x', labelsize=12)
        self.ax[1].set_ylabel('f (kHz)', fontsize=16)
        self.ax[1].set_yticks([0,2000, 4000, 6000, 8000])
        self.ax[1].set_yticklabels(['0', '2', '4', '6', '8'], fontsize=12)
        self.ax[1].set_ylim([1600, 9000])
    

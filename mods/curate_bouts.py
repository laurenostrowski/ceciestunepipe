import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from ceciestunepipe.util.sound import spectral as sp
import noisereduce as nr
from ipywidgets import widgets
from traitlets import CInt, link
from ceciestunepipe.file import bcistructure as et
import warnings
from datetime import datetime, timedelta



def bout_fs_check(bout_pd):
    """ get sample rates for each recording """
    fs_all = []
    # loop through first bout extracted from each unique raw file
    for f in bout_pd['file'].unique():
        fs_all.append(bout_pd.loc[bout_pd[bout_pd['file'] == f].index[0],'sample_rate'])
    # assert same sample rate for all files
    assert(len(np.unique(fs_all)) == 1)
    return fs_all[0]


def epoch_bout_dict_sample_rate_check(bout_pd, sess_par):
    """ get bout dictionaries and sample rates for each epoch """
    fs_all = []
    bout_dicts_all = []
    # loop through epochs
    for this_epoch in np.unique(bout_pd.epoch):
        epoch_struct = et.sgl_struct(sess_par,this_epoch,ephys_software=sess_par['ephys_software'])
        # load bout dictionary
        if sess_par['ephys_software'] == 'sglx':
            bout_dict_path = os.path.join(epoch_struct['folders']['derived'],'bout_dict_ap0.pkl')
        elif sess_par['ephys_software'] == 'oe':
            bout_dict_path = os.path.join(epoch_struct['folders']['derived'],'bout_dict_oe.pkl')
        else:
            raise ValueError('unknown ephys software')
        with open(bout_dict_path, 'rb') as handle:
            bout_dict = pickle.load(handle)
        # store sample rate and bout dictionaires
        fs_all.append(bout_dict['s_f'])
        bout_dicts_all.append(bout_dict)
    # assert same sample rate for all epochs
    assert(len(np.unique(fs_all)) == 1)
    return fs_all[0], bout_dicts_all


def remove_stim_bouts(bout_pd,sess_par):
    """ bout for all epoches where stim overlap is removed """
    bout_pd_in = bout_pd.copy()
    remaining_bouts_all_epoch = []
    # loop through epochs
    for this_epoch in np.unique(bout_pd.epoch):
        # get bouts for this epoch
        this_epoch_bout_pd = bout_pd_in.iloc[np.where(bout_pd_in['epoch'] == this_epoch)]
        # get epoch info
        epoch_struct = et.sgl_struct(sess_par,this_epoch,ephys_software=sess_par['ephys_software'])
        # load synced stim
        stim_pd_path = os.path.join(epoch_struct['folders']['derived'],'stim_pd_ap0.pkl')
        stim_df = pd.read_pickle(stim_pd_path)
        # get stim onsets / offsets
        stim_on_all = stim_df['start_sample']
        stim_off_all = stim_df['end_sample']

        # get bouts that overlap with stim
        stim_bout_list = []
        # loop through stim
        for stim_i in range(len(stim_on_all)):
            # get nearest bout start and stop indexes
            on_before_end = np.where(bout_pd_in.end_sample >= stim_on_all[stim_i])[0]
            off_after_start = np.where(bout_pd_in.start_sample <= stim_off_all[stim_i])[0]
            # if stim happens during bout store bout index
            if (len(on_before_end) > 0) & (len(off_after_start) > 0):
                for store_i in range(on_before_end[0],off_after_start[-1]+1):
                    stim_bout_list.append(store_i)

        # get unique bouts that overlap with stims
        stim_bouts_unique = np.unique(np.array(stim_bout_list))
        # get rid of them
        remaining_bouts = this_epoch_bout_pd.drop(stim_bouts_unique,errors="ignore")
        # store
        remaining_bouts_all_epoch.append(remaining_bouts)

    # regroup across epochs
    bout_pd_updated = pd.concat(remaining_bouts_all_epoch)
    # sort by duration
    bout_pd_updated.sort_values('len_ms',ascending=False,inplace=True)
    bout_pd_updated.reset_index(drop=True,inplace=True)
    
    return bout_pd_updated


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


class Counter(widgets.DOMWidget):
    value = CInt(0)
    value.tag(sync=True)


class VizBout():
    def __init__(self,bouts_pd,recording_sample_rate):
        self.bout = None
        self.bouts_pd = bouts_pd
        self.bout_series = None
        self.is_bout = None
        self.is_call = None
        self.is_confusing = None
        self.bout_counter = None
        self.bout_id = None
        self.buttons = {}
        self.m_pick = None
        self.fig_waveform = None
        self.fig_spectrogram = None
        self.x = None
        self.sxx = None   
        self.fs = recording_sample_rate
        self.sub_sample = 1
        self.init_fig()
        self.init_widget()
        self.show()
                
    def init_fig(self):
#         song_fig = plt.figure(figsize=(10,4))
#         self.wave_plot = song_fig.add_subplot(211)
#         self.spec_plot = song_fig.add_subplot(212)
        self.song_fig, self.ax = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [1,3]}, figsize=(10,4), constrained_layout=True)

    def init_widget(self):
        self.bout_counter = Counter()
        self.is_bout = widgets.Checkbox(description='is bout')
        self.is_call = widgets.Checkbox(description='calls')
        self.is_confusing = widgets.Checkbox(description='confusing')
        self.buttons['Next'] = widgets.Button(description="Next", button_style='info',icon='plus')   
        self.buttons['Prev'] = widgets.Button(description="Prev", button_style='warning',icon='minus')
        self.buttons['Check'] = widgets.Button(description="Song", button_style='success',icon='check')
        self.buttons['Uncheck'] = widgets.Button(description="Noise", button_style='danger',icon='wrong')
        self.buttons['Call'] = widgets.Button(description="Calls")
        [b.on_click(self.button_click) for b in self.buttons.values()]
        top_box = widgets.HBox([self.buttons['Prev'], self.buttons['Next']])
        bottom_box = widgets.HBox([self.buttons['Uncheck'], self.buttons['Check'], self.buttons['Call']])
        button_box = widgets.VBox([top_box, bottom_box])
        self.m_pick = widgets.IntSlider(value=0, min=0, max=self.bouts_pd.index.size-1, step=1, 
                                        description="Bout candidate index")
        control_box = widgets.HBox([button_box,
                                  widgets.VBox([self.is_bout, self.is_confusing, self.is_call]),
                                  widgets.VBox([self.m_pick])])
        link((self.m_pick, 'value'), (self.bout_counter, 'value'))
        self.update_bout()
        self.is_bout.observe(self.bout_checked, names='value')
        self.is_call.observe(self.call_checked, names='value')
        self.is_confusing.observe(self.confusing_checked, names='value')
        self.m_pick.observe(self.slider_change, names='value')
        display(control_box)
        
    def button_click(self, button):        
        self.bout_id = self.bout_counter.value
        curr_bout = self.bout_counter
        if button.description == 'Next':
            curr_bout.value += 1
        elif button.description == 'Prev':
            curr_bout.value -= 1
        elif button.description == 'Song':
            self.bouts_pd.loc[self.bout_id, 'bout_check'] = True
            self.bouts_pd.loc[self.bout_id, 'confusing'] = False
            self.bouts_pd.loc[self.bout_id, 'is_call'] = False
            curr_bout.value += 1
        elif button.description == 'Noise':
            self.bouts_pd.loc[self.bout_id, 'bout_check'] = False
            self.bouts_pd.loc[self.bout_id, 'confusing'] = False
            self.bouts_pd.loc[self.bout_id, 'is_call'] = False
            curr_bout.value += 1
        elif button.description == 'Calls':
            self.bouts_pd.loc[self.bout_id, 'bout_check'] = True
            self.bouts_pd.loc[self.bout_id, 'confusing'] = False
            self.bouts_pd.loc[self.bout_id, 'is_call'] = True
            curr_bout.value += 1
        if curr_bout.value > self.m_pick.max:
            curr_bout.value = 0 
        if curr_bout.value < self.m_pick.min:
            curr_bout.value = self.m_pick.max
    
    def slider_change(self, change):
        self.update_bout()
        self.show()
            
    def bout_checked(self, bc):
        self.bouts_pd.loc[self.bout_id, 'bout_check'] = bc['new']
    
    def call_checked(self, bc):
        self.bouts_pd.loc[self.bout_id, 'is_call'] = bc['new']
    
    def confusing_checked(self, bc):
        self.bouts_pd.loc[self.bout_id, 'confusing'] = bc['new']
    
    def update_bout(self):
        self.bout_id = self.bout_counter.value
        self.bout_series = self.bouts_pd.iloc[self.bout_id]
        self.is_bout.value = bool(self.bout_series['bout_check'])
        self.is_call.value = bool(self.bout_series['is_call'])
        self.is_confusing.value = bool(self.bout_series['confusing'])
        self.x = self.bout_series['waveform'][::self.sub_sample]
        self.sxx = np.flipud(self.bout_series['spectrogram'][::self.sub_sample])
        
    def show(self):
#         time_pts = np.arange(self.x.size) * self.sub_sample / self.fs 
#         self.wave_plot.clear()
#         self.spec_plot.clear()
#         self.wave_plot.plot(time_pts,self.x,c='k')
#         self.spec_plot.imshow(self.sxx,aspect='auto',cmap='inferno')
        [ax.cla() for ax in self.ax]
        # Lowpass filter
        b, a = butter_filt(self.fs, highcut = 300, btype='high')
        x = np.squeeze(self.x[:, 0]) if np.ndim(self.x) > 1 else self.x
        mic_arr_hp = noncausal_filter(x, b, a)
        # Tim Sainburg's noise reduce algorithm
        mic_arr_nr = nr.reduce_noise(x, self.fs, n_std_thresh_stationary=0.5)
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
        self.ax[1].set_yticks([0, 2000, 4000, 6000, 8000])
        self.ax[1].set_yticklabels(['0', '2', '4', '6', '8'], fontsize=12)
        self.ax[1].set_ylim([300, 9000])
        

def give_summary(bpd: pd.DataFrame):
    all_files = np.unique(bpd['file'])
    summary_dict = {f: np.where((bpd['bout_check']==True) & (bpd['file']==f))[0].size for f in all_files}
    return summary_dict


def plot_summary(bpd: pd.DataFrame, start: str, end: str, fig=None, bird=None, date=None):
    all_files = np.unique(bpd['file'])
    summary_dict = {f: np.where((bpd['bout_check']==True) & (bpd['file']==f))[0].size for f in all_files}
    
    # create list of times at 30 minute increments from 'start' to 'end'
    t = datetime.strptime(start, '%H:%M')
    tf = datetime.strptime(end, '%H:%M')
    times = []
    while t <= tf:
        times.append(t.strftime('%H:%M'))
        t += timedelta(minutes=30)
    
    # initialize bouts_count
    bouts_count = np.zeros(len(times), dtype=int)
    
    # insert bouts count into appropriate time slot
    for file_name, bouts in summary_dict.items():
        f = file_name.split('/')[-1]
        match_str = f[:2]+':00' if f[-6:-4]=='01' else f[:2]+':30'
        bouts_count[times.index(match_str)] = bouts

    if fig is None:
        fig = plt.figure(figsize=(5, 5)) # default size

    # plot bar graph
    c = np.arange(1, len(times)+1)
    if len(c) > 1: cmap = ((c - min(c)) / (max(c) - min(c)))
    else: cmap = np.array([1])
    plt.bar(times, bouts_count, color=plt.cm.viridis(cmap[::-1]))
    # if len(c) > 1:
    #     n = len(times)
    #     cmap = plt.cm.viridis(np.linspace(0, 1, n//2))
    #     cmap = np.concatenate([cmap, cmap[::-1]])
        
    # else:
    #     cmap = np.array([1])
    #     cmap = plt.cm.viridis(cmap[::-1])
    # plt.bar(times, bouts_count, color=cmap)
    plt.xlabel('time')
    plt.ylabel('# bouts')
    if bird is not None and date is not None:
        plt.title(f'bird {bird}, session {date}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


def sess_bout_summary(bout_pd: pd.DataFrame, ax_dict: dict=None, bouts_type='curated', min_len_ms=10000,show_hist=True) -> pd.DataFrame:
    ## make and plot a summary of the bird's bout.
    # get lengths of the bouts
    # get estimate timestamps of bouts
    # plot histogram length of bouts
    # histogram of time of bouts
    bout_sel = (bout_pd['valid']==True) & (bout_pd['len_ms'] > min_len_ms);
    if bouts_type=='curated':
        bout_sel = bout_sel & (bout_pd['bout_check']==True) & (bout_pd['confusing']==False);
    
    # len/time? (when do they sing the longest?)
    print('Number of bouts longer than {} secs: {}'.format(min_len_ms//1000, bout_pd.loc[bout_sel].index.size))
    print('Length of all bouts (minutes): {}'.format(np.round(bout_pd.loc[bout_sel, 'len_ms'].values.sum()/60000),5))
    
    if show_hist:
        if ax_dict is None:
            bout_pd.loc[bout_sel].hist(column='len_ms')
    return bout_pd


# Master class
## important: Lo's bout detection algorithm (deep search) doesn't require waveform_edges parameter
##            Zeke's bout detection algorithm (probably running for starling data) adds waveform_edges (ms)
##            to either side of the waveform but does not adjust start_ms, end_ms, start_sample, end_sample, len_ms
##
##            The legacy version is included below.
class TrimBout():
    def __init__(self,bouts_pd, fs):    ## , waveform_edges
        self.bouts_pd = bouts_pd
        self.init_crop_vals = np.column_stack((np.zeros(len(bouts_pd)), # initialize to full bout
                                               bouts_pd['len_ms'].values/1000))    ## + (waveform_edges/500)
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
        self.fs = fs
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
        self.x = self.bout_series['waveform']
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
        b, a = butter_filt(self.fs, highcut = 300, btype='high')
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
        self.ax[1].set_yticks([0, 2000, 4000, 6000, 8000])
        self.ax[1].set_yticklabels(['0', '2', '4', '6', '8'], fontsize=12)
        self.ax[1].set_ylim([300, 9000])


def update_trimmed_bouts(bout_pd, start_s, end_s, fs, fs_ap=None):
    """
     Trim bouts and repopulate entries in bouts dataframe

    Args:
        bout_pd (pandas.DataFrame): bouts dataframe
        trim_bouts: TrimBout class handle
        fs (int): sample rate (Hz)

    Returns:
        pandas.DataFrame: updated bouts dataframe
    """
    bouts_pd_updated = bout_pd.copy()
    
    # Trim waveform
    start_waveform = (start_s * fs).astype(int)
    end_waveform = (end_s * fs).astype(int)
    warnings.simplefilter(action='ignore')
    for i, bout_idx in enumerate(bout_pd.index.tolist()):
        bouts_pd_updated['waveform'][bout_idx] = bout_pd['waveform'][bout_idx][start_waveform[i]:end_waveform[i]]
        
        # Trim auxiliary waveform(s)
        for wave_type in ['male_XLR_waveform', 'fem_XLR_waveform', 'fem_waveform']:
            if wave_type in bout_pd.columns:
                if bout_pd.loc[bout_idx, wave_type] is not None and not np.isnan(bout_pd.loc[bout_idx, wave_type]).all():
                    bouts_pd_updated[wave_type][bout_idx] = bout_pd[wave_type][bout_idx][start_waveform[i]:end_waveform[i]]
    warnings.resetwarnings()
    
    # Verify that sample rates are the same (they should be!!)
    if 'sample_rate' in bout_pd.columns:
        if len(bout_pd.sample_rate.unique()) > 1:
            warnings.warn('More than one USB sample rate detected', UserWarning)
    if 'XLR_sample_rate' in bout_pd.columns:
        if len(bout_pd.XLR_sample_rate.unique()) > 1:
            warnings.warn('More than one XLR sample rate detected', UserWarning)
        if 'sample_rate' in bout_pd.columns:
            for x_sr in bout_pd.XLR_sample_rate.unique():
                for u_sr in bout_pd.sample_rate.unique():
                    if x_sr != u_sr:
                        warnings.warn('Sample rates between USB and XLR mics do not match', UserWarning)
    
    # Update 'start_ms', 'len_ms', and 'end_ms'
    start_ms_trim = (start_s * 1000).astype(int)
    bouts_pd_updated['start_ms'] = (bout_pd.start_ms + start_ms_trim).astype(int)
    bouts_pd_updated['len_ms'] = ((end_waveform - start_waveform) / fs * 1000).astype(int)
    bouts_pd_updated['end_ms'] = (bouts_pd_updated['start_ms'] + bouts_pd_updated['len_ms']).astype(int)
    
    # Recalculate 'start_sample' and 'end_sample'
    bouts_pd_updated['start_sample'] = (bouts_pd_updated['start_ms'] * (fs/1000)).astype(int)
    bouts_pd_updated['end_sample'] = (bouts_pd_updated['end_ms'] * (fs/1000)).astype(int)
    
    # If neural data, update 'start_ms_ap_0' and 'start_sample_ap_0'
    if 'start_ms_ap_0' in bout_pd.columns:
        bouts_pd_updated['start_ms_ap_0'] = (bout_pd.start_ms_ap_0 + start_ms_trim).astype(int)
    if 'start_sample_ap_0' in bout_pd.columns:
        bouts_pd_updated['start_sample_ap_0'] = (bout_pd.start_sample_ap_0 + (start_s * fs_ap)).astype(int)
    
    return bouts_pd_updated


## this works with Zeke's bout detection algorithm
class TrimBoutLegacy():
    def __init__(self,bouts_pd, fs, waveform_edges):
        self.bouts_pd = bouts_pd
        self.init_crop_vals = np.column_stack((np.zeros(len(bouts_pd)), # initialize to full bout
                                               bouts_pd['len_ms'].values/1000 + (waveform_edges/500)))
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
        self.fs = fs
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
        self.x = self.bout_series['waveform']
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


def update_trimmed_bouts_legacy(bout_pd, start_s, end_s, fs, waveform_edges, fs_ap=30000):
    """
     Trim bouts and repopulate entries in bouts dataframe

    Args:
        bout_pd (pandas.DataFrame): bouts dataframe
        trim_bouts: TrimBout class handle
        fs (int): sample rate (Hz)

    Returns:
        pandas.DataFrame: updated bouts dataframe
    """
    bouts_pd_updated = bout_pd.copy()
    
    # Trim waveform
    start_waveform = (start_s * fs).astype(int)
    end_waveform = (end_s * fs).astype(int)
    warnings.simplefilter(action='ignore')
    for i, bout_idx in enumerate(bout_pd.index.tolist()):
        bouts_pd_updated['waveform'][bout_idx] = bout_pd['waveform'][bout_idx][start_waveform[i]:end_waveform[i]]
    warnings.resetwarnings()
    
    # Update 'start_ms', 'len_ms', and 'end_ms'
    start_ms_trim = (start_s * 1000 - waveform_edges).astype(int)
    bouts_pd_updated['start_ms'] = (bout_pd.start_ms + start_ms_trim).astype(int)
    bouts_pd_updated['len_ms'] = ((end_waveform - start_waveform) / fs * 1000).astype(int)
    bouts_pd_updated['end_ms'] = (bouts_pd_updated['start_ms'] + bouts_pd_updated['len_ms']).astype(int)
    
    # Recalculate 'start_sample' and 'end_sample'
    bouts_pd_updated['start_sample'] = (bouts_pd_updated['start_ms'] * (fs/1000)).astype(int)
    bouts_pd_updated['end_sample'] = (bouts_pd_updated['end_ms'] * (fs/1000)).astype(int)
    
    # If neural data, update 'start_ms_ap_0' and 'start_sample_ap_0'
    if 'start_ms_ap_0' in bout_pd.columns:
        bouts_pd_updated['start_ms_ap_0'] = (bout_pd.start_ms_ap_0 + start_ms_trim).astype(int)
    if 'start_sample_ap_0' in bout_pd.columns:
        bouts_pd_updated['start_sample_ap_0'] = (bout_pd.start_sample_ap_0 + (start_s * fs_ap)).astype(int)
    
    return bouts_pd_updated


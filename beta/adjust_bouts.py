import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from ceciestunepipe.util.sound import spectral as sp
import noisereduce as nr

from ipywidgets import widgets
from traitlets import CInt, link
from IPython.display import display, clear_output


# Graphing functions
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

def plot_bout_finch_params(mic_arr, fs, ax):
    """
    Plot microphone array and spectrogram with parameters optimized for zebra finches.

    Args:
        mic_arr (numpy.ndarray): Microphone array
        ax (list of matplotlib.axes): List of axes for plotting
    """
    # Lowpass filter
    b, a = butter_filt(fs, highcut = 1600, btype='high')
    mic_arr_hp = noncausal_filter(mic_arr, b, a)
    
    # Tim Sainburg's noise reduce algorithm
    mic_arr_nr = nr.reduce_noise(mic_arr, fs, n_std_thresh_stationary=0.5)
    
    # Calculate spectrogram
    f, t, sxx = sp.ms_spectrogram(mic_arr_hp.flatten(), fs)
    
    # Graph sonogram
    ax[0].plot(mic_arr_nr.flatten(), 'black')
    ax[0].set_xlim([0, len(mic_arr_nr.flatten())])
    ax[0].set_axis_off()
    
    # Graph spectrogram
    ax[1].pcolormesh(t, f, np.log(sxx), cmap='inferno')
    ax[1].set_xlabel('time (s)', fontsize=16)
    ax[1].set_xlim([t[0], t[-1]])
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].set_ylabel('f (kHz)', fontsize=16)
    ax[1].set_yticks([0,2000, 4000, 6000, 8000])
    ax[1].set_yticklabels(['0', '2', '4', '6', '8'], fontsize=12)
    ax[1].set_ylim([1600, 9000])


# Create a counter object
class Counter(widgets.DOMWidget):
    value = CInt(0)
    value.tag(sync=True)


# Master class
class TrimBoutF():
    def __init__(self, hparams, bouts_pd):
        self.bouts_pd = bouts_pd
        self.bout_series = None
        self.bout_counter = None
        self.bout_id = None
        self.buttons = {}
        self.m_pick = None   # bout candidate index slider
        self.crop = None   # crop range slider
        self.x = None   # mic array
        self.fs = hparams['sample_rate']
        delta = hparams.get('waveform_edges', 0) / 1000
        self.crop_vals = np.column_stack((np.zeros(len(bouts_pd)), bouts_pd['len_ms'].values/1000))
        self.crop_min = self.crop_vals[:,0] - delta # accounting for waveform padding
        self.crop_max = self.crop_vals[:,1] + delta
        self.song_fig, self.ax = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [1,3]}, figsize=(20, 5), constrained_layout=True)
        self.init_widget()
        self.show()
    
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
        # range slider
        self.crop = widgets.FloatRangeSlider(value=self.crop_vals[0,:], # initialize to the first bout
                                             min=self.crop_min[0],
                                             max=self.crop_max[0],
                                             step=0.001,
                                             description='Crop:', 
                                             layout=widgets.Layout(width='100%', height='40px'))  # Adjust dimensions
        self.crop.observe(self.update_crop, names='value')
        # create output
        self.update_bout()
        self.output = widgets.Output() # (layout=widgets.Layout(width='300px')) <-- use this to edit figure width to match crop slider
        display(widgets.VBox([button_box, self.m_pick, self.output, self.crop]))
    
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
    
    def update_crop(self, change):
        self.bout_id = self.bout_counter.value
        self.crop_vals[self.bout_id,:] = change
    
    def update_bout(self):
        self.bout_id = self.bout_counter.value
        self.bout_series = self.bouts_pd.iloc[self.bout_id]
        self.x = self.bout_series['waveform']
        self.crop.min = self.crop_min[self.bout_id]
        self.crop.max = self.crop_max[self.bout_id]
        self.crop.value = list(self.crop_vals[self.bout_id, :])
        
    def show(self):
        [ax.cla() for ax in self.ax]
        plot_bout_finch_params(self.x, self.fs, ax=self.ax)
        plt.close(self.song_fig)
        with self.output:
            clear_output(wait=True)
            display(self.song_fig)


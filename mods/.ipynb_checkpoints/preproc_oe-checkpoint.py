import os
import spikeinterface.sorters as ss # has to go after setting sort environment path
import pandas as pd
import pickle
import traceback
import warnings
import json
import glob
from scipy.io import wavfile
import multiprocessing
N_JOBS_MAX = multiprocessing.cpu_count()
from ceciestunepipe.file import bcistructure as et
from ceciestunepipe.util import fileutil as fu
from ceciestunepipe.util import oeutil as oeu
from ceciestunepipe.util import wavutil as wu
from ceciestunepipe.util import stimutil as st
import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



##### spikeinterface BinDatRecordingExtractor #####
import shutil
import numpy as np
from pathlib import Path
from typing import Union, Optional

from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import read_binary, write_to_binary_dat_format, check_get_traces_args

PathType = Union[str, Path]
DtypeType = Union[str, np.dtype]
ArrayType = Union[list, np.ndarray]
OptionalDtypeType = Optional[DtypeType]
OptionalArrayType = Optional[Union[np.ndarray, list]]


class BinDatRecordingExtractor(RecordingExtractor):
    """
    RecordingExtractor for a binary format

    Parameters
    ----------
    file_path: str or Path
        Path to the binary file
    sampling_frequency: float
        The sampling frequncy
    numchan: int
        Number of channels
    dtype: str or dtype
        The dtype of the binary file
    time_axis: int
        The axis of the time dimension (default 0: F order)
    recording_channels: list (optional)
        A list of channel ids
    geom: array-like (optional)
        A list or array with channel locations
    file_offset: int (optional)
        Number of bytes in the file to offset by during memmap instantiation.
    gain: float or array-like (optional)
        The gain to apply to the traces
    channel_offset: float or array-like
        The offset to apply to the traces
    is_filtered: bool
        If True, the recording is assumed to be filtered
    """
    extractor_name = 'BinDatRecording'
    has_default_locations = False
    has_unscaled = False
    installed = True
    is_writable = True
    mode = "file"
    installation_mesg = ""

    def __init__(self, file_path: PathType, sampling_frequency: float, numchan: int, dtype: DtypeType,
                 time_axis: int = 0, recording_channels: Optional[list] = None,  geom: Optional[ArrayType] = None,
                 file_offset: Optional[float] = 0,
                 gain: Optional[Union[float, ArrayType]] = None,
                 channel_offset: Optional[Union[float, ArrayType]] = None,
                 is_filtered: Optional[bool] = None):
        RecordingExtractor.__init__(self)
        self._datfile = Path(file_path)
        self._time_axis = time_axis
        self._dtype = np.dtype(dtype).name
        self._sampling_frequency = float(sampling_frequency)
        self._numchan = numchan
        self._geom = geom
        self._timeseries = read_binary(self._datfile, numchan, dtype, time_axis, file_offset)

        if is_filtered is not None:
            self.is_filtered = is_filtered
        else:
            self.is_filtered = False

        if recording_channels is not None:
            assert len(recording_channels) <= self._timeseries.shape[0], \
               'Provided recording channels have the wrong length'
            self._channels = recording_channels
        else:
            self._channels = list(range(self._timeseries.shape[0]))

        if len(self._channels) == self._timeseries.shape[0]:
            self._complete_channels = True
        else:
            assert max(self._channels) < self._timeseries.shape[0], "Channel ids exceed the number of " \
                                                                    "available channels"
            self._complete_channels = False

        if geom is not None:
            self.set_channel_locations(self._geom)
            self.has_default_locations = True

        if 'numpy' in str(dtype):
            dtype_str = str(dtype).replace("<class '", "").replace("'>", "")
            dtype_str = dtype_str.split('.')[1]
        else:
            dtype_str = str(dtype)

        if gain is not None:
            self.set_channel_gains(channel_ids=self.get_channel_ids(), gains=gain)
            self.has_unscaled = True

        if channel_offset is not None:
            self.set_channel_offsets(channel_offset)

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'sampling_frequency': sampling_frequency,
                        'numchan': numchan, 'dtype': dtype_str, 'recording_channels': recording_channels,
                        'time_axis': time_axis, 'geom': geom, 'file_offset': file_offset, 'gain': gain,
                        'is_filtered': is_filtered}

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        if self._complete_channels:
            if np.array_equal(channel_ids, self.get_channel_ids()):
                traces = self._timeseries[:, start_frame:end_frame]
            else:
                channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
                if np.all(np.diff(channel_idxs) == 1):
                    traces = self._timeseries[channel_idxs[0]:channel_idxs[0]+len(channel_idxs),
                                              start_frame:end_frame]
                else:
                    # This block of the execution will return the data as an array, not a memmap
                    traces = self._timeseries[channel_idxs, start_frame:end_frame]
        else:
            # in this case channel ids are actually indexes
            traces = self._timeseries[channel_ids, start_frame:end_frame]
        return traces

    @staticmethod
    def write_recording(
        recording: RecordingExtractor,
        save_path: PathType,
        time_axis: int = 0,
        dtype: OptionalDtypeType = None,
        **write_binary_kwargs
    ):
        """
        Save the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor object to be saved in .dat format.
        save_path : str
            The path to the file.
        time_axis : int, optional
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype : dtype
            Type of the saved data. Default float32.
        **write_binary_kwargs: keyword arguments for write_to_binary_dat_format() function
        """
        write_to_binary_dat_format(recording, save_path, time_axis=time_axis, dtype=dtype,
                                   **write_binary_kwargs)


##### zeke built oe recording extractor #####
class oeRecordingExtractor(BinDatRecordingExtractor):
    extractor_name = 'oeContinuousRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'

    _ttl_events = None  # The ttl events
    _t_0 = None # time computed naively (n/s_f_0)
    _t_prime = None # time synchronized to a pattern ('master') reference
    _s_f_0 = None # measured samplin rate (using the syn signal)
    _syn_chan_id = None # digital channel for signal id (for nidaq; automatic last channel in lf/ap streams)
    _dig = None # the digital signal
    _start_sample = None # start sample from the beginning of the run
    _start_t = None # start t (absolute in the machine)
    
    _meta_dict = None # dictionary with metadata of the recording
    _chan_pd = None # pandas dataframe with channel ids
    
    def __init__(self, rec_path: str, processor, dtype: str = 'int16', syn_chan_id=0):
        # dtype should come from the meta but for now its ok
        # rec_path is the path to the recording
        self._meta_dict = get_rec_meta(rec_path)
        self._chan_pd = build_chan_info_pd(self._meta_dict)
        self._s_f_0 = get_oe_sample_rate(self._meta_dict)
        
        cont_path = os.path.join(rec_path, 'continuous', processor, 'continuous.dat')
        
        n_chan = self._chan_pd['recorded'].sum()
        BinDatRecordingExtractor.__init__(self, cont_path, self._s_f_0, n_chan, np.int16)
        
        self._chan_names = np.array(self._chan_pd .loc[self._chan_pd ['recorded']==True, 'name'])
        
def get_oe_cont_recording(exp_struct: dict, epoch:str):
    epoch_path = exp_struct['folders']['oe']
    node_path = oeu.get_default_node(exp_struct, epoch)
    rec_path = oeu.get_default_recording(node_path)
    cont_path = oeu.get_default_continuous(rec_path)
    
    default_processor = os.path.split(cont_path)[-1]
    #print(default_processor)
    oe_recording = oeRecordingExtractor(rec_path, default_processor)
    
    return oe_recording

def get_oe_epoch_path(exp_struct: dict, epoch:str):
    epoch_path = exp_struct['folders']['oe']
    node_path = oeu.get_default_node(exp_struct, epoch)
    return node_path
        

##### zeke oe preprocessing functions #####
def list_oe_epochs(exp_struct):
    sess_path = os.path.join(exp_struct['folders']['oe'])
    epoch_list = [os.path.split(f.path)[-1] for f in os.scandir(sess_path) if f.is_dir()]
    return epoch_list

def list_nodes(epoch_path):
    return [f.path for f in os.scandir(epoch_path) if f.is_dir()]

def list_experiments(node_path):
    return [f.path for f in os.scandir(node_path) if f.is_dir()]

def list_recordings(experiment_path):
    return [f.path for f in os.scandir(experiment_path) if f.is_dir()]

def list_processors(signal_path):
    return [f.path for f in os.scandir(signal_path) if f.is_dir()]

def get_rec_meta(rec_path):
    rec_meta_path = os.path.join(rec_path, 'structure.oebin')
    with open(rec_meta_path, 'r') as meta_file:
        meta = json.load(meta_file)
    return meta

def get_continous_files_list(rec_path, processor='Rhythm_FPGA-100.0'):
    cont_raw_list = glob.glob(os.path.join(rec_path, 'continuous', processor, 'continuous.dat'))
    print(cont_raw_list)
    return cont_raw_list

def oe_list_bin_files(epoch_path):
    return glob.glob(os.path.join(epoch_path, 'experiment*.dat'))

def get_default_node(exp_struct, epoch, rec_index=0):
    # get the first rec node, the first experiment, and ith index of recording
    r_path = os.path.join(os.path.join(exp_struct['folders']['oe'], epoch))
    node = list_nodes(r_path)[0]
    
    r_path = os.path.join(r_path, node)
    experiment = list_experiments(r_path)[0]
    
    return r_path

def get_default_recording(node_path):
    experiment = list_experiments(node_path)[0]
    r_path = os.path.join(node_path, experiment)
    
    recording = list_recordings(r_path)[0]
    r_path = os.path.join(r_path, recording)
    return r_path

def get_default_continuous(rec_path):
    processor = list_processors(os.path.join(rec_path, 'continuous'))[0]
    r_path = os.path.join(rec_path, processor)
    return r_path

def get_oe_sample_rate(rec_meta_dict: dict) -> float:
    return float(rec_meta_dict['continuous'][0]['sample_rate'])


def build_chan_info_pd(oe_meta_dict: dict, processor_order: int=0) -> pd.DataFrame:
    # read all channels names, numbers, and whether they were recorded
    rec_chan_meta = oe_meta_dict['continuous'][processor_order]['channels']
    
    all_chan_meta = []
    for i, a_chan_meta in enumerate(rec_chan_meta):
        all_chan_meta.append({'number': i,
                              'recorded': 1,
                             'name': a_chan_meta['channel_name'],
                             'gain': float(a_chan_meta['bit_volts'])})
        
    all_chan_pd = pd.DataFrame(all_chan_meta)
    return all_chan_pd

def find_chan_order(chan_info_pd: pd.DataFrame, chan_name: str) -> int:
    recorded_block_pd = chan_info_pd[chan_info_pd['recorded']==1]
    recorded_block_pd.reset_index(inplace=True, drop=True)
    return recorded_block_pd[recorded_block_pd['name']==chan_name].index[0]
        
        
##### zeke preprocessing steps #####     
def chan_to_wav(single_bin_path, chan_name, oe_meta_dict, wav_path,skip_wav=True):
    # get the file n of channels
    chan_info_pd = build_chan_info_pd(oe_meta_dict)
    n_channels = chan_info_pd['recorded'].sum()
    logger.info(n_channels)
    chan_pos = find_chan_order(chan_info_pd, chan_name)
    logger.info(chan_pos)
    # read the file
    bin_fp = np.memmap(single_bin_path, dtype='<i2', mode='r').reshape(-1, n_channels)
    
    #save as wav
    sample_rate = int(get_oe_sample_rate(oe_meta_dict))
    logger.info('writing wave file {}'.format(wav_path))
#     wavfile.write(wav_path, sample_rate, bin_fp.T[chan_pos])
    stream_out = bin_fp.T[chan_pos]
    wav_s_f = wu.save_wav(stream_out, sample_rate, wav_path, skip_wav=skip_wav)
    return stream_out


def preprocess_run(sess_par: dict, exp_struct: dict, epoch:str, do_sync_to_stream=None, skip_wav=True) -> dict:
    # get the recording files
    # dump the microphone file into a wav file
    # that's it for now
    
    logger.info('PREPROCESSING sess {} | epoch {}'.format(sess_par['sess'], epoch))
    
    raw_folder = exp_struct['folders']['oe']
    epoch_path = os.path.join(raw_folder, epoch)
    node_path = get_default_node(exp_struct, epoch)
    rec_path = get_default_recording(node_path)
    cont_path = os.path.join(get_default_continuous(rec_path), 'continuous.dat')
    
    rec_meta = get_rec_meta(rec_path)
    sample_rate = int(get_oe_sample_rate(rec_meta))
    
    logger.info('getting the recording file ' + cont_path)
    
    # get the rig parameters
    # get the mic channel name in the channels recorded
    # make the folder for the derived data
    # toss the wav file in there
    
    rig_par = et.get_rig_par(exp_struct)
    
    mic_ch_name = rig_par['chan']['adc']['microphone_0']
    
    derived_path = os.path.join(exp_struct['folders']['derived'], epoch)
    fu.makedirs(derived_path)
    wav_path = os.path.join(derived_path, 'wav_mic.wav')
    logger.info('get microphone from ch {}'.format(wav_path))
    chan_to_wav(cont_path, mic_ch_name, rec_meta, wav_path, skip_wav=skip_wav)
    
    if 'adc_list' in sess_par:
        adc_list = sess_par['adc_list']
    else:
        adc_list = []
    if len(adc_list) > 0:
        adc_list = sess_par['adc_list']
        logger.info('Getting adc channel(s) {}'.format(adc_list))
        for adc_label in adc_list:     
            adc_path = os.path.join(derived_path,adc_label,'.wav')
            adc_ch_name = rig_par['chan']['adc'][adc_label]
            chan_to_wav(cont_path, adc_ch_name, rec_meta, adc_path, skip_wav=skip_wav)
    
    if 'stim_sess' in sess_par:
        if len(sess_par['stim_sess']) > 0:
            stim_list = sess_par['stim_list']
        else:
            stim_list = []
    else:
        stim_list = []
    if len(stim_list) > 0:
        logger.info('Getting stimulus channel(s) {}'.format(stim_list))       
        stim_path = os.path.join(derived_path,'wav_stim.wav')
        stim_ch_name = rig_par['chan']['adc']['wav_stim']
        chan_to_wav(cont_path, stim_ch_name, rec_meta, stim_path, skip_wav=skip_wav)
        if 'wav_syn' in stim_list:
            logger.info('Getting the onset/offset of stimuli from the {} extracted analog channel'.format('wav_syn'))
            syn_path = os.path.join(derived_path,'wav_syn.wav')
            syn_ch_name = rig_par['chan']['adc']['wav_syn']
            wav_sync_stream = chan_to_wav(cont_path, syn_ch_name, rec_meta, syn_path, skip_wav=skip_wav)
            sine_ev, sine_ttl, t_ttl = st.get_events_from_sine_sync(wav_sync_stream, sample_rate, step_ms=100)

            sine_ttl_path = os.path.join(derived_path, 'wav_stim_sync_sine_ttl.npy')
            sine_ttl_t_path = os.path.join(derived_path, 'wav_stim_sync_sine_ttl_t.npy')
            sine_ev_path = os.path.join(derived_path, 'wav_stim_sync_sine_ttl_evt.npy')
            np.save(sine_ttl_path, sine_ttl)
            np.save(sine_ttl_t_path, t_ttl)
            np.save(sine_ev_path, sine_ev)
            logger.info('saved onset/offset of trial events from the sine wave in ' + sine_ev_path)
    
    return rec_meta


## sequentially process all runs of the sessions
def preprocess_session(sess_par: dict,force_redo=False,skip_wave=True):
    logger.info('pre-process all runs of sess ' + sess_par['sess'])
    # get exp struct
    sess_struct = et.get_exp_struct(sess_par['bird'], sess_par['sess'], sort=sess_par['sort'], ephys_software='oe')
     # check if derived data exists - don't run unless force redo
    run_this_preproc = True
    if os.path.exists(sess_struct['folders']['derived']):
        print('derived data folder exists..')
        if not force_redo:
            run_this_preproc = False
            print('no force redo..')
    if run_this_preproc:
        print('preprocessing..')
        # list the epochs
        sess_epochs = list_oe_epochs(sess_struct)
        logger.info('found epochs: {}'.format(sess_epochs))
        # preprocess all epochs
        epoch_dict_list = []
        for i_ep, epoch in enumerate(sess_epochs):
            try:
                exp_struct = et.sgl_struct(sess_par, epoch)
                one_epoch_dict = preprocess_run(sess_par, sess_struct, epoch, skip_wav=skip_wave)
                epoch_dict_list.append(one_epoch_dict)
            except Exception as exc:
                warnings.warn('Error in epoch {}'.format(epoch), UserWarning)
                logger.info(traceback.format_exc)
                logger.info(exc)
                logger.info('Session {} epoch {} could not be preprocessed'.format(sess_par['sess'], epoch))
                
                
##### zeke run oe spike sort kilosort function #####
def run_spikesort(recording_extractor: oeRecordingExtractor, 
                  logger: logging.Logger,
                  sort_pickle_path: str,
                  tmp_dir: str, 
                  grouping_property: str=None,
                  sorting_method: str='kilosort3',
                  n_jobs_bin: int=N_JOBS_MAX,
                  chunk_mb=8192,restrict_to_gpu=None,
                  parallel=False,force_redo=False,
                  **sort_kwargs):

    logger.info("Grouping property: {}".format(grouping_property))
    logger.info("sorting method: {}".format(sorting_method))
    
    if sorting_method == "kilosort2":
        sort_tmp_dir = os.path.join(tmp_dir, 'tmp_ks2')
    elif sorting_method == "kilosort3":
        sort_tmp_dir = os.path.join(tmp_dir, 'tmp_ks3')
    else: 
        print('Only have kilosort 2/3 implemented')
        breakme
         
    logger.info('Sorting tmp dir {}'.format(sort_tmp_dir))
        
    if force_redo is False:
        try:
            spk_clu = np.load(os.path.join(sort_tmp_dir, 'spike_clusters.npy'))
            logger.info('Found previous sort in tmp folder {} and force_redo is false.'.format(sort_tmp_dir))
            run_sort_flag = False
            sort = 'previously run'
        except:
            logger.info('Previous sort not found, sorting')
            run_sort_flag = True
    else:
        run_sort_flag = True
      
    if run_sort_flag:
        if restrict_to_gpu is not None:
            logger.info('Will set visible gpu devices {}'.format(restrict_to_gpu))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(restrict_to_gpu)
        if sorting_method == "kilosort2":
            try:            
                sort = ss.run_kilosort2(
                    recording_extractor,
                    car=True,
                    output_folder=sort_tmp_dir,
                    parallel=parallel,
                    verbose=True,
                    grouping_property=grouping_property,
                    chunk_mb=chunk_mb,
                    n_jobs_bin=n_jobs_bin,
                    **sort_kwargs)
            except:
                sort = np.nan
        elif sorting_method == "kilosort3":
            try:
                sort = ss.run_kilosort3(
                    recording_extractor,
                    car=True,
                    output_folder=sort_tmp_dir,
                    parallel=parallel,
                    verbose=True,
                    grouping_property=grouping_property,
                    chunk_mb=chunk_mb,
                    **sort_kwargs)
            except:
                sort = np.nan
        
        try:
            np.isnan(sort)
        except: # save sort
            logger.info("Saving sort {}".format(sort_pickle_path))
            with open(sort_pickle_path, "wb") as output:
                pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
            logger.info("Sorting output saved to {}".format(sort_pickle_path)) 

            # save sort again with all that processed data
            sort_temp_pickle_path = sort_pickle_path + '.dump.pkl'
            logger.info("Saving sort {}".format(sort_temp_pickle_path))
            sort.dump_to_pickle(sort_temp_pickle_path)
        
    logger.info('done sorting')
        
    return sort


##### zeke script for oe sync #####
def bout_dict_from_pd(bout_pd: pd.DataFrame, all_syn_dict: dict) -> dict:
    s_f_wav = all_syn_dict['wav']['s_f']
    
    start_ms = bout_pd['start_ms'].values
    len_ms = bout_pd['len_ms'].values
    
    ### fill in the bout to be compatible with the boutpd from the neuropix
    # but all clocks are the same
    
    bout_dict = {
            's_f': s_f_wav, # s_f used to get the spectrogram
            's_f_nidq': all_syn_dict['nidq']['s_f'],
            's_f_ap_0': all_syn_dict['ap_0']['s_f'],
            'start_ms': start_ms,
            'len_ms': len_ms,
           'start_sample_naive': ( start_ms * s_f_wav * 0.001).astype(np.int64),
           'start_sample_nidq': ( start_ms * s_f_wav * 0.001).astype(np.int64),
            'start_sample_wav': ( start_ms * s_f_wav * 0.001).astype(np.int64),
    }
    bout_dict['start_ms_ap_0'] = start_ms
    bout_dict['start_sample_ap_0'] = bout_dict['start_sample_naive']
    
    # complete for compatibility with the sglx/neuropix datasets
    bout_pd['start_sample_ap_0'] =  bout_pd['start_sample']
    bout_pd['start_sample_naive'] =  bout_pd['start_sample']
    bout_pd['start_ms_ap_0'] = bout_pd['start_ms']
    return bout_dict
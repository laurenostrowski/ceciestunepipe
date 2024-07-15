import logging
import os
import time
import datetime

from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

from spikeextractors import RecordingExtractor
from .readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args


# types for variables in function prototypes
# https://docs.python.org/3/library/typing.html
from typing import Union
intOrNone = Union[int, None]

logger = logging.getLogger(
    'ceciestunepipe.util.spikeextractors.extractors.spikeglxrecordingextractor.spikeglxrecordingextractor')


class SpikeGLXRecordingExtractor(RecordingExtractor):
    extractor_name = 'SpikeGLXRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    # error message when not installed
    installation_mesg = "To use the SpikeGLXRecordingExtractor run:\n\n pip install mtscomp\n\n"

    _ttl_events = None  # The ttl events
    _t_0 = None # time computed naively (n/s_f_0)
    _t_prime = None # time synchronized to a pattern ('master') reference
    _s_f_0 = None # measured samplin rate (using the syn signal)
    _syn_chan_id = None # digital channel for signal id (for nidaq; automatic last channel in lf/ap streams)
    _dig = None # the digital signal
    _start_sample = None # start sample from the beginning of the run
    _start_t = None # start t (absolute in the machine)

    def __init__(self, file_path: str, dtype: str = 'int16', syn_chan_id=0):
        RecordingExtractor.__init__(self)
        self._npxfile = Path(file_path)
        self._basepath = self._npxfile.parents[0]

        assert dtype in [
            'int16', 'float'], "'dtype' can be either 'int16' or 'float'"
        self._dtype = dtype
        # Gets file type: 'imec0.ap', 'imec0.lf' or 'nidq'
        assert 'imec0.ap' in self._npxfile.name or 'imec0.lf' in self._npxfile.name or 'nidq' in self._npxfile.name, \
            "'file_path' can be an imec0.ap, imec.lf, or nidq file"
        assert 'bin' in self._npxfile.name, "The 'npx_file should be either the 'ap' or the 'lf' bin file."
        if 'imec0.ap' in str(self._npxfile):
            lfp = False
            ap = True
            self.is_filtered = True
        elif 'imec0.lf' in str(self._npxfile):
            lfp = True
            ap = False
        else:
            lfp = False
            ap = False
        aux = self._npxfile.stem.split('.')[-1]
        if aux == 'nidq':
            self._ftype = aux
        else:
            self._ftype = self._npxfile.stem.split('.')[-2] + '.' + aux

        # Metafile
        self._metafile = self._basepath.joinpath(self._npxfile.stem+'.meta')
        if not self._metafile.exists():
            raise Exception("'meta' file for '"+self._ftype +
                            "' traces should be in the same folder.")
        # Read in metadata, returns a dictionary
        meta = readMeta(self._npxfile)
        self._meta = meta

        # Traces in 16-bit format
        if '.cbin' in self._npxfile.name:  # compressed binary format used by IBL
            try:
                import mtscomp
            except:
                raise Exception(self.installation_mesg)
            self._raw = mtscomp.Reader()
            self._raw.open(self._npxfile, self._npxfile.with_suffix('.ch'))
        else:
            # [chanList, firstSamp:lastSamp+1]
            self._raw = makeMemMapRaw(self._npxfile, meta)

        # sampling rate and ap channels
        self._sampling_frequency = SampRate(meta)
        tot_chan, ap_chan, lfp_chan, locations = _parse_spikeglx_metafile(
            self._metafile)
        if ap:
            if ap_chan < tot_chan:
                self._channels = list(range(int(ap_chan)))
                self._timeseries = self._raw[0:ap_chan, :]
            else:
                # OriginalChans(meta).tolist()
                self._channels = list(range(int(tot_chan)))
        elif lfp:
            if lfp_chan < tot_chan:
                self._channels = list(range(int(lfp_chan)))
                self._timeseries = self._raw[0:lfp_chan, :]
            else:
                self._channels = list(range(int(tot_chan)))
        else:
            # nidq
            self._channels = list(range(int(tot_chan)))
            self._timeseries = self._raw

        # locations
        if len(locations) > 0:
            self.set_channel_locations(locations)

        # get gains
        if meta['typeThis'] == 'imec':
            gains = GainCorrectIM(self._timeseries, self._channels, meta)
        elif meta['typeThis'] == 'nidq':
            gains = GainCorrectNI(self._timeseries, self._channels, meta)

        # set gains - convert from int16 to uVolt
        self.set_channel_gains(self._channels, gains*1e6)
        self._kwargs = {'file_path': str(
            Path(file_path).absolute()), 'dtype': dtype}

        self.set_syn_chan_id(syn_chan_id)

    def set_syn_chan_id(self, syn_chan_id=0):
        self._syn_chan_id = syn_chan_id

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_start(self):
        self._start_sample = self._meta['firstSample']

        # get the file creation time in seconds from the metadata; it's the best we have so far
        self._start_t = get_creation_time(self)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, dtype=None):
        channel_idxs = np.array(
            [self.get_channel_ids().index(ch) for ch in channel_ids])
        if np.all(channel_ids == self.get_channel_ids()):
            recordings = self._timeseries[:, start_frame:end_frame]
        else:
            if np.all(np.diff(channel_idxs) == 1):
                recordings = self._timeseries[channel_idxs[0]                                              :channel_idxs[0]+len(channel_idxs), start_frame:end_frame]
            else:
                # This block of the execution will return the data as an array, not a memmap
                recordings = self._timeseries[channel_idxs,
                                              start_frame:end_frame]
        if dtype is not None:
            assert dtype in [
                'int16', 'float'], "'dtype' can be either 'int16' or 'float'"
        else:
            dtype = self._dtype
        if dtype == 'int16':
            return recordings
        else:
            gains = np.array(self.get_channel_gains())[channel_idxs]
            return recordings * gains[:, None]

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        logger.info('getting ttl events, chan {}'.format(channel_id))
        channel = [channel_id]
        dw = 0
        dig = ExtractDigital(self._raw, firstSamp=start_frame, lastSamp=end_frame, dwReq=dw, dLineList=channel,
                             meta=self._meta)
        dig = np.squeeze(dig).astype(int)
        diff_dig = np.diff(dig)

        rising = np.where(diff_dig > 0)[0] + start_frame
        falling = np.where(diff_dig < 0)[0] + start_frame

        ttl_frames = np.concatenate((rising, falling))
        ttl_states = np.array([1] * len(rising) + [-1] * len(falling))
        sort_idxs = np.argsort(ttl_frames)

        self._ttl_events = tuple(
            [ttl_frames[sort_idxs], ttl_states[sort_idxs]])
        return ttl_frames[sort_idxs], ttl_states[sort_idxs]

    def save_ttl_events(self, file_path: str)->np.array:
         # save ttl_events as a npy file 
         # go from the tuple to the array 
         pass

    def get_ttl_traces(self, start_frame=None, end_frame=None, channel_list=np.arange(8), ignore_existing=True):
        logger.info('getting ttl traces, chan {}'.format(channel_list))
        if (self._dig is None) or (ignore_existing ==True):
            # get all of the digital traces
            dw = 0
            if end_frame is None:
                end_frame = self._raw.shape[-1]
            if start_frame is None:
                start_frame = 0
            dig = ExtractDigital(self._raw, firstSamp=start_frame, lastSamp=end_frame, dwReq=dw, dLineList=channel_list,
                             meta=self._meta)
            self._dig = np.squeeze(dig).astype(np.short)
        
        return self._dig

    def get_effective_sf(self, start_frame: intOrNone = None, end_frame: intOrNone = None,
                         force_ttl: bool = False) -> tuple:
        if (self._ttl_events is None) or force_ttl:
            syn_chan_id = self._syn_chan_id
            self.get_ttl_events(start_frame, end_frame, syn_chan_id)

        syn_ttl = self._ttl_events
        s_f_arr = compute_sf(syn_ttl)

        n_samples = self.get_traces().shape[-1]

        self._s_f_0 = np.mean(s_f_arr)
        self._t_0 = np.arange(n_samples)/self._s_f_0

        return self._s_f_0, self._t_0, self._ttl_events

    def syn_to_pattern(self, t_0_pattern: np.array, ttl_edge_tuple_pattern: tuple,
                       force_ttl: bool = False) -> tuple:

        # get the times from the pattern signal corresponding to the samples right at the edges
        # those have to be the exact ones in the t prime of the current stream
        # anything in between will be a pieceweise linearly interpolated time

        # get t0, sampling rate and ttl edges of the current stream
        s_f_0, t_0, ttl_edge_tuple = self.get_effective_sf(force_ttl=force_ttl)

        # check the number of edges, they should match
        n_edges = ttl_edge_tuple[0].size
        n_edges_pattern = ttl_edge_tuple_pattern[0].size
        if n_edges != n_edges_pattern:
            # If the signals don't have the same number of edges there may be an error, better stop and debug
            raise ValueError(
                'Number of edges in the syn ttl events of pattern and target dont match')

        # the 'actual' times at the edges of the syn signal
        t_pattern_edge = t_0_pattern[ttl_edge_tuple_pattern[0]]
        # the interpolation function. fill_value='extrapolate' allows extrapolation from zero and until the last time stamp
        # careful, this could lead to negative time, but it is the correct way to do it.
        t_interp_f = interp1d(ttl_edge_tuple[0], t_pattern_edge,
                              assume_sorted=True, fill_value='extrapolate')

        n_samples = t_0.size
        t_prime = t_interp_f(np.arange(n_samples))
        self._t_prime = t_prime
        return t_prime

    def read_tprime(self):
        return self._t_prime

    def syn_to_sgl_rec_exctractor(self, extractor_pattern: RecordingExtractor, force_ttl: bool = False):

        # get the pattern ttl edges and t at them
        s_f_pattern, t_pattern, ttl_edge_tuple_pattern = extractor_pattern.get_effective_sf(
            force_ttl=force_ttl)

        # sync the tprime of this object to the pattern
        self.syn_to_pattern(
            t_pattern, ttl_edge_tuple_pattern, force_ttl=force_ttl)

    # def get_syn_events(self, start_frame=None, end_frame=None, channel_id=0):
    #     # SYN is in one of the digital channels in the nidaq, or
    #     # in the last channel in the lf/ap

    #     if self._meta['typeThis'] == 'imec':
    #         # check if there was a syn channel
    #         # get the syn channel within the bounds
    #         # get the onsets, offsets
    #         # return in get_ttl_events style


def _parse_spikeglx_metafile(metafile):
    tot_channels = None
    ap_channels = None
    lfp_channels = None
    x_pitch = 21
    y_pitch = 20

    locations = []
    with Path(metafile).open() as f:
        for line in f.readlines():
            if 'nSavedChans' in line:
                tot_channels = int(line.split('=')[-1])
            if 'snsApLfSy' in line:
                ap_channels = int(line.split('=')[-1].split(',')[0].strip())
                lfp_channels = int(line.split(',')[-2].strip())
            if 'imSampRate' in line:
                fs = float(line.split('=')[-1])
            if 'snsShankMap' in line:
                map = line.split('=')[-1]
                chans = map.split(')')[1:]
                for chan in chans:
                    chan = chan[1:]
                    if len(chan) > 0:
                        x_pos = int(chan.split(':')[1])
                        y_pos = int(chan.split(':')[2])
                        locations.append([x_pos*x_pitch, y_pos*y_pitch])
    return tot_channels, ap_channels, lfp_channels, locations


def compute_sf(ttl: tuple) -> np.array:
    ttl_arr = np.array(ttl)

    # get all the diffs betwenn edge ups and edge donws
    all_diff_arr = np.concatenate(
        [np.diff(ttl_arr[0, ttl_arr[1] == j]) for j in [-1, 1]])

    return all_diff_arr


def get_creation_time(sglx_recording):
    bin_file_path = sglx_recording._npxfile
    
    ## file creation time is volatile (esp after copying in windows), 
    ## compute the start form the last modified and file duration
    ## this kind of sucks, but it's the best we have
    bin_end_tstamp = os.path.getmtime(bin_file_path)
    bin_end_str = datetime.datetime.fromtimestamp(bin_end_tstamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    logger.debug('file end {}'.format(bin_end_str))
    
    bin_start_tstamp = bin_end_tstamp - float(sglx_recording._meta['fileTimeSecs'])
    bin_start_str = datetime.datetime.fromtimestamp(bin_start_tstamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    logger.debug('file start {}'.format(bin_start_str))
    
    return bin_start_tstamp
import more_itertools as mit
import peakutils
import warnings
import traceback
import numpy as np
import pandas as pd
import pickle
import logging
import os
import glob
import sys
import traceback
import datetime
import time

from scipy.io import wavfile
from tqdm.auto import tqdm

from ceciestunepipe.util import fileutil as fu
from ceciestunepipe.util import wavutil as wu
from ceciestunepipe.util.sound import spectral as sp
from ceciestunepipe.util.sound import temporal as st


logger = logging.getLogger('ceciestunepipe.util.sound.boutsearch')


def read_wav_chan(wav_path: str, chan_id: int = 0, return_int16=True) -> tuple:
    # return_int16 makes a safe casting to int16 to not confuse librosa et al in the following steps
    s_f, x = wavfile.read(wav_path, mmap=True)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if return_int16:
        if x.dtype == 'int32':
            y = (x[:, chan_id] >> 16).astype(np.int16)
            #y = (z/np.max(np.abs(z)) * 32000).astype(np.int16)
        elif x.dtype == 'int16':
            y = x[:, chan_id]
            #y = (x[:, chan_id]/np.max(np.abs(x[:, chan_id])) * 32000).astype(np.int16)
        else:
            raise NotImplementedError(
                'wav file is neither int16 nor int32 and I dont know how to convert to int16 yet')
    else:
        y = x[:, chan_id]

    return s_f, y

def read_npy_chan(wav_path: str, chan_id: int = 0, return_int16=True) -> tuple:
    s_f, x = wu.read_wav_chan(wav_path, chan_id=chan_id, skip_wav=True)

    if return_int16:
        if x.dtype == 'int32':
            y = (x >> 16).astype(np.int16)
            #y = (z/np.max(np.abs(z)) * 32000).astype(np.int16)
        elif x.dtype == 'int16':
            y = x
            #y = (x[:, chan_id]/np.max(np.abs(x[:, chan_id])) * 32000).astype(np.int16)
        else:
            raise NotImplementedError(
                'wav file is neither int16 nor int32 and I dont know how to convert to int16 yet')
    else:
        y = x

    return s_f, y

def sess_file_id(f_path):
    n = int(os.path.split(f_path)[1].split('-')[-1].split('.')[0])
    return n


def sample_rate_from_wav(wav_path):
    #sample_rate, x = wavfile.read(wav_path)
    sample_rate, x = wu.read_wav_chan(wav_path)
    return sample_rate


class BoutParamsUnpickler(pickle.Unpickler):
    # hparams during the search in boutsearch contains functions that are saved in the pickle.
    # Loading the pickle naively will search for those functions in the __main__ module and will fail.
    # This custom pickle loader will interject and replace the functions with the ones in the boutsearch module.
    # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    def find_class(self, module, name):
        if name == 'read_wav_chan':
            return read_wav_chan
        elif name == 'sess_file_id':
            return sess_file_id
        else:
            return super().find_class(module, name)


class BoutParamsPickler(pickle.Pickler):
    # hparams during the search in boutsearch contains functions that are saved in the pickle.
    # Loading the pickle naively will search for those functions in the __main__ module and will fail.
    # This custom pickle loader will interject and replace the functions with the ones in the boutsearch module.
    # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    def find_class(self, module, name):
        if name == 'read_wav_chan':
            logger.info('bout pickler')
            from ceciestunepipe.util.sound import boutsearch as bs
            return bs.read_wav_chan
        elif name == 'sess_file_id':
            return sess_file_id
        else:
            return super().find_class(module, name)


default_hparams = {  # default parameters work well for starling
    # spectrogram
    'num_freq': 1024,  # 1024# how many channels to use in a spectrogram #
    'preemphasis': 0.97,
    'frame_shift_ms': 5,  # step size for fft
    'frame_length_ms': 10,  # 128 # frame length for fft FRAME SAMPLES < NUM_FREQ!!!
    'min_level_db': -55,  # minimum threshold db for computing spe
    'ref_level_db': 110,  # reference db for computing spec
    'sample_rate': None,  # sample rate of your data

    # spectrograms
    'mel_filter': False,  # should a mel filter be used?
    'num_mels': 1024,  # how many channels to use in the mel-spectrogram
    'fmin': 500,  # low frequency cutoff for mel filter
    'fmax': 12000,  # high frequency cutoff for mel filter

    # spectrogram inversion
    'max_iters': 200,
    'griffin_lim_iters': 20,
    'power': 1.5,

    # Added for the searching
    # function for loading the wav_like_stream (has to returns fs, ndarray)
    'read_wav_fun': read_wav_chan,
    # function for extracting the file id within the session
    'file_order_fun': sess_file_id,
    # Minimum length of supra_threshold to consider a 'syllable' (ms)
    'min_segment': 30,
    # Minmum distance between groups of syllables to consider separate bouts (ms)
    'min_silence': 3000,
    'min_bout': 3000,  # min bout duration (ms)
    'peak_thresh_rms': 0.55,  # threshold (rms) for peak acceptance,
    'thresh_rms': 0.25,  # threshold for detection of syllables
    # threshold for acceptance of mean rms across the syllable (relative to rms of the file)
    'mean_syl_rms_thresh': 0.3,
    'max_bout': 180000,  # exclude bouts too long
    # threshold for n of len_ms/peaks (typycally about 2-3 syllable spans
    'l_p_r_thresh': 100,

    # get number of ms before and after the edges of the bout for the waveform sample
    'waveform_edges': 1000,

    # extension for saving the auto found files
    'bout_auto_file': 'bout_auto.pickle',
    # extension for manually curated files (coming soon)
    'bout_curated_file': 'bout_checked.pickle',
}


def gimmepower(x, hparams):
    #x = x/np.max(np.abs(x)) * 32000
    #y = x.astype(np.int16)
    s = sp.rosa_spectrogram(x.flatten(), hparams)
    f = np.arange(hparams['num_freq']) / \
        hparams['num_freq']*0.5*hparams['sample_rate']
    s = s[(f > hparams['fmin']) & (f < hparams['fmax']), :]
    f = f[(f > hparams['fmin']) & (f < hparams['fmax'])]
    p = s.sum(axis=0)
    return p, f, s


def get_on_segments(x, thresh=0, min_segment=20, pk_thresh=0, mean_rms_thresh=0):
    on = np.where(x > thresh)[0]
    on_segments = [list(group) for group in mit.consecutive_groups(on)]
    logger.debug('On segments {}'.format(len(on_segments)))
    if len(on_segments) > 0:
        hi_segments = np.vstack([np.array([o[0], o[-1]]) for o in on_segments
                                 if ((np.max(x[o]) > pk_thresh) and
                                     (st.rms(x[o]) > mean_rms_thresh))])
    else:
        hi_segments = np.array([])
    if len(hi_segments) > 0:
        long_enough_segments = hi_segments[(
            np.diff(hi_segments) >= min_segment).flatten(), :]
    else:
        long_enough_segments = np.array([])

    logger.debug('good segments shape {}'.format(long_enough_segments.shape))
    return long_enough_segments


def merge_near_segments(on_segs, min_silence=200):
    # merge all segments distant less than min_silence
    # need to have at least two bouts
    if on_segs.shape[0] < 2:
        logger.debug('Less than two zero segments, nothing to possibly merge')
        long_segments = on_segs
    else:
        of_on = on_segs.flatten()[1:]
        silence = np.diff(of_on)[::2]
        long_silence = np.where(silence > min_silence)[0]
        if(long_silence.size == 0):
            logger.debug('No long silences found, all is one big bout')
        of_keep = np.append((on_segs[long_silence, 1]), on_segs[-1, 1])
        on_keep = np.append(on_segs[0, 0], on_segs[long_silence + 1, 0])
        long_segments = np.vstack([on_keep, of_keep]).T
    return long_segments


def get_the_bouts(x, spec_par_rosa, loaded_p=None):
    #
    if loaded_p is not None:
        p = loaded_p
        logger.debug('loaded p with shape {}'.format(loaded_p.shape))
    else:
        logger.debug('Computing power')
        p, _, _ = gimmepower(x, spec_par_rosa)

    logger.debug('Finding on segments')
    threshold = spec_par_rosa['thresh_rms'] * st.rms(p)
    pk_threshold = spec_par_rosa['peak_thresh_rms'] * st.rms(p)
    mean_rms_threshold = spec_par_rosa['mean_syl_rms_thresh'] * st.rms(p)
    step_ms = spec_par_rosa['frame_shift_ms']
    min_syl = spec_par_rosa['min_segment'] // step_ms
    min_silence = spec_par_rosa['min_silence'] // step_ms
    min_bout = spec_par_rosa['min_bout'] // step_ms
    max_bout = spec_par_rosa['max_bout'] // step_ms

    syllables = get_on_segments(
        p, threshold, min_syl, pk_threshold, mean_rms_threshold)
    logger.debug('Found {} syllables'.format(syllables.shape[0]))

    logger.debug(
        'Merging segments with silent interval smaller than {} steps'.format(min_silence))
    bouts = merge_near_segments(syllables, min_silence=min_silence)
    logger.debug('Found {} bout candidates'.format(bouts.shape[0]))

    if bouts.shape[0] > 0:
        long_enough_bouts = bouts[((np.diff(bouts) >= min_bout) & (
            np.diff(bouts) < max_bout)).flatten(), :]
        logger.debug('Removed shorter/longer than [{} ;{}], {} candidates left'.format(min_bout, max_bout,
                                                                                       long_enough_bouts.shape[0]))
    else:
        long_enough_bouts = bouts
    power_values = [p[x[0]:x[1]] for x in long_enough_bouts]

    return long_enough_bouts, power_values, p, syllables


def get_bouts_in_file(file_path, hparams, loaded_p=None, return_error_code=False):
    # path of the wav_file
    # h_params from the rosa spectrogram plus the parameters:
    #     'read_wav_fun': load_couple, # function for loading the wav_like_stream (has to returns fs, ndarray)
    #     'min_segment': 30, # Minimum length of supra_threshold to consider a 'syllable'
    #     'min_silence': 200, # Minmum distance between groups of syllables to consider separate bouts
    #     'bout_lim': 200, # same as min_dinscance !!! Clean that out!
    #     'min_bout': 250, # min bout duration
    #     'peak_thresh_rms': 2.5, # threshold (rms) for peak acceptance,
    #     'thresh_rms': 1 # threshold for detection of syllables

    # Decide and see if it CAN load the power
    logger.info('Getting bouts for file {}'.format(file_path))
    #print('file {}...'.format(file_path))
    sys.stdout.flush()
    #logger.debug('s_f {}'.format(s_f))

    # Get the bouts. If loaded_p is none, it will copute it
    try:
        s_f, wav_i = hparams['read_wav_fun'](file_path)
        hparams['sample_rate'] = s_f
        the_bouts, the_p, all_p, all_syl = get_the_bouts(
            wav_i, hparams, loaded_p=loaded_p)
        error_code = 0  # success

    except Exception as e:
        warn_msg = 'Error getting bouts for file ' + file_path
        warnings.warn(warn_msg)
        logger.info('error message was {}'.format(e))
        #print('Un {}'.format(file_path))
        sys.stdout.flush()
        traceback.print_exc()
        # return empty DataFrame
        the_bouts = np.empty(0)
        wav_i = np.empty(0)
        all_p = np.empty(0)
        error_code = 1  # there was an error

    if the_bouts.size > 0:
        step_ms = hparams['frame_shift_ms']
        pk_dist = hparams['min_segment']
        bout_pd = pd.DataFrame(the_bouts * step_ms,
                               columns=['start_ms', 'end_ms'])
        bout_pd['start_sample'] = bout_pd['start_ms'] * (s_f//1000)
        bout_pd['end_sample'] = bout_pd['end_ms'] * (s_f//1000)

        bout_pd['p_step'] = the_p
        # the extrema over the file
        bout_pd['rms_p'] = st.rms(all_p)
        bout_pd['peak_p'] = bout_pd['p_step'].apply(np.max)
        # check whether the peak power is larger than hparams['peak_thresh_rms'] times the rms through the file
        bout_pd['bout_check'] = bout_pd.apply(lambda row:
                                              (row['peak_p'] > hparams['peak_thresh_rms']
                                               * row['rms_p']),
                                              axis=1)
        bout_pd['file'] = file_path
        bout_pd['len_ms'] = bout_pd.apply(
            lambda r: r['end_ms'] - r['start_ms'], axis=1)

        syl_pd = pd.DataFrame(
            all_syl * step_ms, columns=['start_ms', 'end_ms'])
        bout_pd['syl_in'] = bout_pd.apply(lambda r:
                                          syl_pd[(syl_pd['start_ms'] >= r['start_ms']) &
                                                 (syl_pd['start_ms'] <= r['end_ms'])].values,
                                          axis=1)
        bout_pd['n_syl'] = bout_pd['syl_in'].apply(len)
        # get all the peaks larger than the threshold(peak_thresh_rms * rms)
        bout_pd['peaks_p'] = bout_pd.apply(lambda r: peakutils.indexes(r['p_step'],
                                                                       thres=hparams['peak_thresh_rms'] *
                                                                       r['rms_p'] /
                                                                       r['p_step'].max(
        ),
            min_dist=pk_dist//step_ms),
            axis=1)
        bout_pd['n_peaks'] = bout_pd['peaks_p'].apply(len)
        bout_pd['l_p_ratio'] = bout_pd.apply(
            lambda r: np.nan if r['n_peaks'] == 0 else r['len_ms'] / (r['n_peaks']), axis=1)

        try:
            delta = int(hparams['waveform_edges'] *
                        hparams['sample_rate'] * 0.001)
        except KeyError:
            delta = 0

        bout_pd['waveform'] = bout_pd.apply(
            lambda df: wav_i[df['start_sample'] - delta: df['end_sample'] + delta], axis=1)

    else:
        bout_pd = pd.DataFrame()

    if return_error_code:
        logger.info('returning empty dataframe for file {}'.format(file_path))
        return bout_pd, wav_i, all_p, error_code
    else:
        return bout_pd, wav_i, all_p


def get_bouts_in_long_file(file_path, hparams, loaded_p=None, chunk_size=50000000):

    # Decide and see if it CAN load the power
    logger.info('Getting bouts for long file {}'.format(file_path))
    #print('tu vieja file {}'.format(file_path))
    sys.stdout.flush()
    #logger.debug('s_f {}'.format(s_f))

    # Get the bouts. If loaded_p is none, it will copute it
    s_f, wav_i = hparams['read_wav_fun'](file_path)
    hparams['sample_rate'] = s_f

    n_chunks = int(np.ceil(wav_i.shape[0]/chunk_size))
    wav_chunks = np.array_split(wav_i, n_chunks)
    logger.info('splitting file into {} chunks'.format(n_chunks))

    chunk_start_sample = 0
    bouts_pd_list = []
    for i_chunk, wav_chunk in tqdm(enumerate(wav_chunks), total=n_chunks):
        # get the bouts for a chunk
        # offset the starts to the beginning of the chunk
        # recompute the beginning of the next chunk
        the_bouts, the_p, all_p, all_syl = get_the_bouts(
            wav_chunk, hparams, loaded_p=loaded_p)
        chunk_offset_ms = int(1000 * chunk_start_sample / s_f)

        # make one bouts pd dataframe for the chunk
        if the_bouts.size > 0:
            step_ms = hparams['frame_shift_ms']
            pk_dist = hparams['min_segment']
            bout_pd = pd.DataFrame(
                the_bouts * step_ms + chunk_offset_ms, columns=['start_ms', 'end_ms'])
            bout_pd['start_sample'] = bout_pd['start_ms'] * (s_f//1000)
            bout_pd['end_sample'] = bout_pd['end_ms'] * (s_f//1000)

            bout_pd['p_step'] = the_p
            # the extrema over the file
            bout_pd['rms_p'] = st.rms(all_p)
            bout_pd['peak_p'] = bout_pd['p_step'].apply(np.max)
            # check whether the peak power is larger than hparams['peak_thresh_rms'] times the rms through the file
            bout_pd['bout_check'] = bout_pd.apply(lambda row:
                                                  (row['peak_p'] > hparams['peak_thresh_rms']
                                                   * row['rms_p']),
                                                  axis=1)
            bout_pd['file'] = file_path
            bout_pd['len_ms'] = bout_pd.apply(
                lambda r: r['end_ms'] - r['start_ms'], axis=1)

            syl_pd = pd.DataFrame(
                all_syl * step_ms + + chunk_offset_ms, columns=['start_ms', 'end_ms'])
            bout_pd['syl_in'] = bout_pd.apply(lambda r:
                                              syl_pd[(syl_pd['start_ms'] >= r['start_ms']) &
                                                     (syl_pd['start_ms'] <= r['end_ms'])].values,
                                              axis=1)
            bout_pd['n_syl'] = bout_pd['syl_in'].apply(len)
            # get all the peaks larger than the threshold(peak_thresh_rms * rms)
            bout_pd['peaks_p'] = bout_pd.apply(lambda r: peakutils.indexes(r['p_step'],
                                                                           thres=hparams['peak_thresh_rms']*r['rms_p']/r['p_step'].max(
            ),
                min_dist=pk_dist//step_ms),
                axis=1)
            bout_pd['n_peaks'] = bout_pd['peaks_p'].apply(len)
            bout_pd['l_p_ratio'] = bout_pd.apply(
                lambda r: np.nan if r['n_peaks'] == 0 else r['len_ms'] / (r['n_peaks']), axis=1)

            # ### refer the starts, ends to the beginning of the chunk
            # delta_l = -1*delta - chunk_start_sample
            # delta_r = delta - cunk_start_sample
            # bout_pd['waveform'] = bout_pd.apply(lambda df: wav_chunk[df['start_sample'] + delta_l: df['end_sample'] + delta_r], axis=1)

        else:
            bout_pd = pd.DataFrame()

        chunk_start_sample += wav_chunk.size
        bouts_pd_list.append(bout_pd)

    all_bout_pd = pd.concat(bouts_pd_list)
    all_bout_pd.reset_index(inplace=True, drop=True)
    # get all the waveforms
    if not all_bout_pd.empty:
        try:
            delta = int(hparams['waveform_edges'] *
                        hparams['sample_rate'] * 0.001)
        except KeyError:
            delta = 0

        all_bout_pd['waveform'] = all_bout_pd.apply(lambda df: wav_i[df['start_sample'] - delta: df['end_sample'] + delta],
                                                    axis=1)

        all_bout_pd['confusing'] = True
        all_bout_pd['bout_check'] = False

    return all_bout_pd, wav_i


def apply_files_offset(sess_pd, hparams):
    # append a column with the absolute timing of the start-end in the day of recording
    # all files assumed to have same length
    logger.debug('Applying file offsets')
    s_f, one_wav = hparams['read_wav_fun'](sess_pd.loc[0]['file'])
    file_len = one_wav.shape[0]
    file_len_ms = file_len // (s_f // 1000)

    logger.debug('File len is {}s'.format(file_len_ms*.001))
    sess_pd['i_file'] = sess_pd['file'].apply(hparams['file_order_fun'])
    sess_pd['start_abs'] = sess_pd['start_ms'] + \
        sess_pd['i_file'] * file_len_ms
    sess_pd['end_abs'] = sess_pd['end_ms'] + sess_pd['i_file'] * file_len_ms

    sess_pd['start_abs_sample'] = sess_pd['start_sample'] + \
        sess_pd['i_file'] * file_len
    sess_pd['end_abs_sample'] = sess_pd['end_sample'] + \
        sess_pd['i_file'] * file_len
    return sess_pd


def get_bouts_session(raw_folder, proc_folder, hparams, force_p_compute=False):
    logger.info('Going for the bouts in all the files of the session {}'.format(
        os.path.split(raw_folder)[-1]))
    logger.debug('Saving all process files to {}'.format(proc_folder))

    sess_files = glob.glob(os.path.join(raw_folder, '*.wav'))
    sess_files.sort()
    logger.debug('Found {} files'.format(len(sess_files)))

    all_bout_pd = [pd.DataFrame()]
    for i, raw_file_path in enumerate(sess_files):
        logger.debug('raw file path {}'.format(raw_file_path))
        _, file_name = os.path.split(raw_file_path)
        p_file_path = os.path.join(
            proc_folder, file_name.split('.wav')[0] + '.npy')
        #logger.debug('p file path {}'.format(p_file_path))

        if force_p_compute:
            loaded_p = None
        else:
            try:
                loaded_p = np.load(p_file_path)
            except FileNotFoundError:
                logger.debug('Power file not found, computing')
                loaded_p = None
            except AttributeError:
                logger.debug('No power file path entered, computing')
                loaded_p = None

        try:
            bout_pd, _, p = get_bouts_in_file(
                raw_file_path, hparams, loaded_p=loaded_p)
            bout_pd['file_p'] = p_file_path
            if loaded_p is None:
                logger.debug('Saving p file {}'.format(p_file_path))
                np.save(p_file_path, p)

            all_bout_pd.append(bout_pd)
        except Exception as exc:
            e = sys.exc_info()[0]
            logger.warning(
                'Error while processing {}: {}'.format(raw_file_path, e))
            logger.info(traceback.format_exc)
            logger.info(exc)

    big_pd = pd.concat(all_bout_pd, axis=0, ignore_index=True, sort=True)

    # apply some refinements, filter those that have good waveforms and get spectrograms
    if (hparams['file_order_fun'] is not None) and big_pd.index.size > 0:
        big_pd = apply_files_offset(big_pd, hparams)
        big_pd = cleanup(big_pd)
        logger.info('getting spectrograms')
        big_pd['spectrogram'] = big_pd['waveform'].apply(
            lambda x: gimmepower(x, hparams)[2])

    out_file = os.path.join(proc_folder, hparams['bout_auto_file'])
    big_pd.to_pickle(out_file)
    fu.chmod(out_file, 0o777)
    logger.info('Saved all to {}'.format(out_file))

    return big_pd


def get_epoch_bouts(i_path: str, hparams: dict) -> pd.DataFrame:
    epoch_bout_pd = get_bouts_in_long_file(i_path, hparams)[0]

    i_folder = os.path.split(i_path)[0]
    epoch_bouts_path = os.path.join(i_folder, hparams['bout_auto_file'])
    hparams_pickle_path = os.path.join(i_folder, 'bout_search_params.pickle')

    save_bouts_params_dict(hparams, hparams_pickle_path)
    logger.info('saving bouts pandas to ' + epoch_bouts_path)
    epoch_bout_pd.to_pickle(epoch_bouts_path)
    fu.chmod(epoch_bouts_path, 0o777)

    #epoch_bout_pd = pd.DataFrame()
    return epoch_bout_pd


def save_bouts_params_dict(hparams: dict, hparams_pickle_path: str):
    logger.info('saving bout detect parameters dict to ' + hparams_pickle_path)
    saveparams = hparams.copy()

    with open(hparams_pickle_path, 'wb') as fh:
        if(saveparams['read_wav_fun'].__name__ == 'read_wav_chan'):
            saveparams['read_wav_fun'] = read_wav_chan
        if(saveparams['file_order_fun'].__name__ == 'sess_file_id'):
            saveparams['file_order_fun'] = sess_file_id
        pickler = BoutParamsPickler(fh)
        pickler.dump(saveparams)
    fu.chmod(hparams_pickle_path, 0o777)


def load_bouts_params_dict(hparams_pickle_path: str):
    logger.info('loading detect parameters dict from ' + hparams_pickle_path)
    with open(hparams_pickle_path, 'rb') as fh:
        unpickler = BoutParamsUnpickler(fh)
        hparams = unpickler.load()


def cleanup(bout_pd: pd.DataFrame) -> pd.DataFrame:
    # check for empty waveforms (how woudld THAT happen???)
    bout_pd['valid_waveform'] = bout_pd['waveform'].apply(
        lambda x: (False if x.size == 0 else True))

    # valid is & of all the validated criteria
    bout_pd['valid'] = bout_pd['valid_waveform']

    # drop not valid and reset index
    bout_pd.drop(bout_pd[bout_pd['valid'] == False].index, inplace=True)
    bout_pd.reset_index(drop=True, inplace=True)
    return bout_pd


def alsa_file_timestamp(file_path: str):
    logger.debug(file_path)
    path_parts = fu.get_path_parts(file_path)
    date_str = path_parts[-3]
    # alsa time is hh-mm-ss-x where x is the nth file, and files are split every 30min
    # see /home/finch/scripts/record_bird.sh for instance
    # the name of the file can be HH-MM-SS-xx if split recording, or HH-MM-SS if non-split.
    # if non split, just warn and add a 01
    time_str_list = path_parts[-1].split('.wav')[0].split('-')
    if len(time_str_list) == 3:
        logger.warning('Non split recording detected in file {}'.format(file_path))
        time_str_list.append('01')


    time_str = '{}:{}:{}'.format(*(time_str_list[:-1]))

    datetime_str = '{}_{}'.format(date_str, time_str)
    logger.debug('datetime_str ' + datetime_str)
    strp_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d_%H:%M:%S")

    extra_seconds = 1800 * (int(time_str_list[-1]) - 1)
    strp_time += datetime.timedelta(seconds=extra_seconds)
    #logger.info('strp_time {}'.format(strp_time))
    return strp_time

def alsa_bout_time_stamps(bout_df: pd.DataFrame) -> pd.DataFrame:
    bout_df['t_stamp'] = bout_df.apply(lambda x: alsa_file_timestamp(x['file']) + datetime.timedelta(milliseconds=x['start_ms']), 
                         axis=1)
    return bout_df

def sess_bout_summary(bout_df: pd.DataFrame, ax_dict: dict = None, bouts_type='curated', min_len_ms=7000,
                      rec_software='alsa') -> pd.DataFrame:
    # make and plot a summary of the bird's bout.
    # get lengths of the bouts
    # get estimate timestamps of bouts
    # plot histogram length of bouts
    # histogram of time of bouts
    
    # if implemented, get the timestamp of every bout in the dataframe
    if rec_software in ['alsa']:
        alsa_bout_time_stamps(bout_df)

    bout_sel = (bout_df['valid'] == True) & (bout_df['len_ms'] > min_len_ms)
    if bouts_type == 'curated':
        bout_sel = bout_sel & (bout_df['bout_check'] == True) & (
            bout_df['confusing'] == False)

    # len/time? (when do they sing the longest?)
    logger.info('Number of bouts: {}'.format(bout_df.loc[bout_sel].index.size))
    logger.info('Length of all bouts (minutes): {}'.format(
        bout_df.loc[bout_sel, 'len_ms'].values.sum()/60000))

    if ax_dict is None:
        bout_df.loc[bout_sel].hist(column='len_ms')
    return bout_df

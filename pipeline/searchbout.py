from xmlrpc.client import boolean
from pickle import Unpickler
from builtins import FileExistsError, FileNotFoundError, NotImplementedError
import more_itertools as mit
import warnings
import traceback
import numpy as np
import pandas as pd
import pickle
import logging
import os
import glob
import socket
import sys
import datetime

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed
from scipy.io import wavfile
from tqdm.auto import tqdm

from ceciestunepipe.util.sound import spectral as sp
from ceciestunepipe.util.sound import temporal as st

from ceciestunepipe.util import fileutil as fu

from ceciestunepipe.file import bcistructure as et
from ceciestunepipe.util.sound import boutsearch as bs

from ceciestunepipe.mods import bout_detection_mf as bd_mf

logger = logging.getLogger('ceciestunepipe.pipeline.searchbout')


class BoutParamsUnpickler(pickle.Unpickler):
    # hparams during the search in boutsearch contains functions that are saved in the pickle.
    # Loading the pickle naively will search for those functions in the __main__ module and will fail.
    # This custom pickle loader will interject and replace the functions with the ones in the boutsearch module.
    # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    def find_class(self, module, name):
        if name == 'read_wav_chan':
            from ceciestunepipe.util.sound import boutsearch as bs
            return bs.read_wav_chan
        elif name == 'sess_file_id':
            from ceciestunepipe.util.sound import boutsearch as bs
            return bs.sess_file_id
        else:
            return super().find_class(module, name)


def get_all_day_bouts(sess_par: dict, hparams: dict, 
                      ephys_software='alsa', 
                      file_ext='wav', 
                      n_jobs: int = 12, 
                      save=True,
                      deep_search=False) -> pd.DataFrame:

    logger.info('Will search for bouts through all session {}, {}'.format(
        sess_par['bird'], sess_par['sess']))
    exp_struct = et.get_exp_struct(
        sess_par['bird'], sess_par['sess'], ephys_software=ephys_software)

    # get all the paths to the wav files of the day
    if ephys_software == 'alsa':
        if not file_ext=='wav':
            raise ValueError('alsa files should be wav extension and NOT {}, ainnit?'.format(file_ext))
        # data comes from the raw_data
        source_folder = exp_struct['folders'][ephys_software]
        wav_path_list = glob.glob(os.path.join(source_folder, '*.wav'))

    elif ephys_software in ['sglx', 'oe']:
        # data comes from the derived_data
        source_folder = exp_struct['folders']['derived']
        wav_path_list = et.get_sgl_files_epochs(
            source_folder, file_filter='*wav_mic.{}'.format(file_ext))

    else:
        raise NotImplementedError(
            'Dont know how to deal with {} recording software'.format(ephys_software))

    logger.info('getting {} files from {}'.format(file_ext, source_folder))
    wav_path_list.sort()
    logger.info('Found {} files'.format(len(wav_path_list)))

    def get_file_bouts(i_path):
        if ephys_software == 'alsa': 
            bpd = bs.get_bouts_in_file(i_path, hparams)[0]
            #bpd = bs.alsa_bout_time_stamps(bpd)
            return bpd
        else:
            bpd = pd.DataFrame()
            try:
                if deep_search:
                    bpd = bd_mf.get_bouts_in_long_file(i_path, hparams)
                else:
                    bpd, _ = bs.get_bouts_in_long_file(i_path, hparams)
            except Exception:
                logger.info('Error in file ' + i_path)
                logger.info(traceback.format_exc())
            return bpd

    # Go parallel through all the paths in the day, get a list of all the pandas dataframes for each file
    if n_jobs > 1:
        sess_pd_list = Parallel(n_jobs=n_jobs, verbose=100, backend=None)(
            delayed(get_file_bouts)(i) for i in wav_path_list)
    else:
        sess_pd_list = [get_file_bouts(i) for i in wav_path_list]

    #sess_pd_list = [bs.get_bouts_in_file(i, hparams)[0] for i in wav_path_list]
    # update the sample rate
    # quick and dirty, find the first one that is possible
    for i_path in wav_path_list:
        try:
            hparams['sample_rate'] = bs.sample_rate_from_wav(i_path)
            break
        except:
            logger.info('could not get rate from file {}'.format(i_path))

    # concatenate the file, filter by 'good' waveform, get spectrogram and reindex
    sess_bout_pd = pd.concat(sess_pd_list)
    # before doing check that it's not empty
    if sess_bout_pd.index.size > 0:
        sess_bout_pd = bs.cleanup(sess_bout_pd)
        if sess_bout_pd.index.size > 0:
            logger.info('getting spectrograms')
            sess_bout_pd['spectrogram'] = sess_bout_pd['waveform'].apply(
                lambda x: bs.gimmepower(x, hparams)[2])
        else:
            logger.info('Bouts dataframe came out empty after cleaning up')
    else:
        logger.info('Bouts dataframe came out empty after search')

    # finally, save
    sess_bout_pd.reset_index(drop=True, inplace=True)
    if save:
        save_auto_bouts(sess_bout_pd, sess_par, hparams,
                        software=ephys_software, bout_file_key='bout_auto_file')
    return sess_bout_pd


def save_auto_bouts(sess_bout_pd, sess_par, hparams, software='alsa', bout_file_key='bout_auto_file'):

    if software == 'alsa':
        exp_struct = et.get_exp_struct(sess_par['bird'], sess_par['sess'],
                                       ephys_software='alsa')
        sess_bouts_dir = os.path.join(exp_struct['folders']['derived'],
                                      'bouts_ceciestunepipe')

    elif software in ['sglx', 'oe']:
        exp_struct = et.get_exp_struct(sess_par['bird'], sess_par['sess'],
                                       ephys_software='bouts_' + software)
        sess_bouts_dir = exp_struct['folders']['derived']

    else:
        raise NotImplementedError(
            'Not know how to save bouts for software ' + software)
    sess_bouts_path = os.path.join(sess_bouts_dir, hparams[bout_file_key])
    hparams_file_name = 'bout_search_params.pickle'

    hparams_pickle_path = os.path.join(
        sess_bouts_dir, hparams_file_name)

    fu.makedirs(sess_bouts_dir, exist_ok=True, mode=0o777)
    logger.info('saving bouts pandas to ' + sess_bouts_path)
    sess_bout_pd.to_pickle(sess_bouts_path)
    fu.chmod(sess_bouts_path, 0o777)

    logger.info('saving bout detect parameters dict to ' + hparams_pickle_path)
    bs.save_bouts_params_dict(hparams, hparams_pickle_path)
    # with open(hparams_pickle_path, 'wb') as fh:
    #    pickle.dump(hparams, fh)
    #fu.chmod(hparams_pickle_path, 0o777)


def bout_to_wav(a_bout: pd.Series, sess_par, hparams, dest_dir):
    file_name = '{}_{}_{}.wav'.format(sess_par['sess'],
                                      os.path.split(
                                          a_bout['file'])[-1].split('.wav')[0],
                                      a_bout['start_ms'])
    file_path = os.path.join(dest_dir, file_name)
    x = a_bout['waveform']

    wavfile.write(file_path, hparams['sample_rate'], x)
    return file_path


def bouts_to_wavs(sess_bout_pd, sess_par, hparams, dest_dir):
    # make the dest_dir if does not exist
    logger.info('Saving all session bouts to folder ' + dest_dir)
    fu.makedirs(dest_dir, exist_ok=True, mode=0o777)
    # write all the motifs to wavs
    sess_bout_pd.apply(lambda x: bout_to_wav(
        x, sess_par, hparams, dest_dir), axis=1)
    # write the hparams as pickle
    hparams_pickle_path = os.path.join(
        dest_dir, 'bout_search_params_{}.pickle'.format(sess_par['sess']))

    logger.info('saving bout detect parameters dict to ' + hparams_pickle_path)
    with open(hparams_pickle_path, 'wb') as fh:
        pickle.dump(hparams, fh)


def has_bouts_file(bird: str, sess: str, ephys_software: str, derived_folder: str = 'bouts_ceciestunepipe',
                   bout_type: str = 'bout_auto_file') -> bool:

    exp_struct = et.get_exp_struct(bird, sess, ephys_software=ephys_software)
    bouts_folder = os.path.join(
        exp_struct['folders']['derived'], derived_folder)
    hparams_file_path = os.path.join(bouts_folder, 'bout_search_params.pickle')

    try:
        with open(hparams_file_path, 'rb') as fh:
            unpickler = BoutParamsUnpickler(fh)
            hparams = unpickler.load()

        bouts_auto_file_path = os.path.join(bouts_folder, hparams[bout_type])
        has_bout_file = True if os.path.isfile(bouts_auto_file_path) else False

    except FileNotFoundError:
        logger.info('Search/bouts file not found in {}'.format(bouts_folder))
        has_bout_file = False
    return has_bout_file


def load_bouts(bird: str, sess: str, ephys_software: str, derived_folder: str = 'bouts_ceciestunepipe', bout_file_key='bout_auto_file') -> pd.DataFrame:

    exp_struct = et.get_exp_struct(bird, sess, ephys_software=ephys_software)
    bouts_folder = os.path.join(
        exp_struct['folders']['derived'], derived_folder)
    logger.info(bouts_folder)

    # try loading the params, bouts file
    # if exist, return that one, otherwise return None
    hparams_file_path = os.path.join(bouts_folder, 'bout_search_params.pickle')

    # start with none to return None by default
    bouts_pd = None  # this one starts as empty dataframe
    hparams = None

    try:
        #read_wav_chan = copy_func(bs.read_wav_chan)
        with open(hparams_file_path, 'rb') as fh:
            unpickler = BoutParamsUnpickler(fh)
            hparams = unpickler.load()

        bouts_auto_file_path = os.path.join(
            bouts_folder, hparams[bout_file_key])
        # load. It is important to reset index because the manual curation requires unique indexing
        try:
            bouts_pd = pd.read_pickle(
                bouts_auto_file_path).reset_index(drop=True)
        except FileNotFoundError:
            logger.info('Bout parameters file not found {}'.format(
                bouts_auto_file_path))

    except FileNotFoundError:
        logger.info('Search parameters file not found {}'.format(
            hparams_file_path))

    return hparams, bouts_pd


def load_bouts_alsa(bird: str, sess: str, bout_df_type: str) -> pd.DataFrame:
    
    bouts_folder = os.path.join('/mnt/sphere/speech_bci/derived_data', bird, sess, 'bout_detection_threshold')
    bouts_pd = None
    
    # interpret inputs
    bout_df_types = ['auto', 'checked', 'curated']
    if bout_df_type not in bout_df_types:
        raise ValueError(f"Invalid bout dataframe type. Expected one of: {bout_df_types}")
    
    # load bouts
    try:
        bouts_pd = pd.read_pickle(os.path.join(bouts_folder,'bouts_'+bout_df_type+'.pickle'))
    except FileNotFoundError:
        logger.info('Bout dataframe not found {}'.format())
    
    return bouts_pd


def search_bird_bouts(bird: str, sess_list: list, hparams: dict, ephys_software: str = 'alsa', n_jobs=4, force=False) -> int:
    logger.info(
        'Getting all bouts in bird {} sessions {}'.format(bird, sess_list))

    # check if bouts exist for the sessions for this bird
    for sess in sess_list:
        sess_par = {'bird': bird, 'sess': sess}
        if (not force) and has_bouts_file(bird, sess, ephys_software=ephys_software):
            logger.info(
                'Bird {} already had a bouts file in sess {}'.format(bird, sess))
        else:
            logger.info(
                'Will search bouts for Bird {} - sess {}'.format(bird, sess))
            try:
                sess_bout_pd = get_all_day_bouts(
                    sess_par, hparams, ephys_software=ephys_software, n_jobs=n_jobs)
                logger.info('Found {} bout candidates'.format(
                    sess_bout_pd.index.size))
            except:
                logger.info('Error on bird {} - sess {}'.format(bird, sess))
                traceback.print_exc()

    return 0


def all_bird_bouts_search(bird: str, days_lookup: int, hparams: dict, ephys_software: str = 'alsa', n_jobs=4, force=False, do_today=False) -> int:
    # get all bouts in a bird for the last days_lookup days
    from_date = datetime.date.today() - datetime.timedelta(days=days_lookup)
    to_date = datetime.date.today() if do_today else datetime.date.today() - \
        datetime.timedelta(days=1)

    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    logger.info('Getting all bouts for bird {} from date {} onward'.format(
        bird, from_date_str))
    if do_today:
        logger.info('Including today')

    sess_arr = np.array(et.list_sessions(
        bird, section='raw', ephys_software=ephys_software))
    sess_arr.sort()

    if sess_arr.size==0:
        logger.info('No sessions for bird {}; will skip search_birds_bout'.format(bird))
        search_result = None
    else:
        sess_arr = sess_arr[(sess_arr >= from_date_str)
                            & (sess_arr <= to_date_str)]
        do_sess_list = list(sess_arr)

        search_result = search_bird_bouts(
            bird, do_sess_list, hparams, ephys_software=ephys_software, n_jobs=n_jobs, force=force)
    
    return search_result


def get_birds_bouts(birds_list: list, days_lookup: int, hparams: dict, ephys_software: str = 'alsa', n_jobs=4, force=False, do_today=False) -> int:
    logger.info('Getting all bouts for birds {} for the last {} days'.format(
        birds_list, days_lookup))

    for bird in birds_list:
        search_result = all_bird_bouts_search(
            bird, days_lookup, hparams, ephys_software=ephys_software, n_jobs=n_jobs, force=force, do_today=do_today)
    return search_result


def get_starlings_alsa_bouts(days_lookup: int, n_jobs: int = 4, force: boolean = False, do_today=False, 
                             birds_list: list=[]) -> int:
    # get all birds
    if len(birds_list) > 0:
        logger.info('Entered bird list {}'.format(birds_list))
    else:
        birds_list = et.list_birds(section='raw', ephys_software='alsa')

    # apply a filter to get the starlings
    starling_list = [b for b in birds_list if b.split('_')[0] == 's']

    # get all_sessions for the bird list
    starling_hparams = bs.default_hparams
    get_birds_bouts(starling_list, days_lookup, starling_hparams,
                    ephys_software='alsa', n_jobs=n_jobs, force=force, do_today=do_today)


def get_one_day_bouts(bird, sess):
    starling_hparams = bs.default_hparams
    search_result = search_bird_bouts(
        bird, [sess], starling_hparams, ephys_software='alsa', n_jobs=12)


def read_session_bouts(bird: str, sess: str, recording_software='alsa', curated: boolean = False) -> pd.DataFrame:
    # get the location of the sess, parameters pd.
    # if has spectrograms do nothing, otherwise compute
    # reindex/reset index
    # reset the manual labels
    #exp_struct = et.get_exp_struct(bird, sess, ephys_software=recording_software)
    #derived_folder = exp_struct['folders']['derived']
    #logger.info('Looking for params, bout pickle file in folder {}'.format(derived_folder))

    bout_file_key = 'bout_curated_file' if curated else 'bout_auto_file'
    hparams, bout_pd = load_bouts(bird, sess, recording_software,
                                  derived_folder='bouts_ceciestunepipe', bout_file_key=bout_file_key)
    # this could be something (none or empty) or none.
    bout_pd = pd.DataFrame() if bout_pd is None else bout_pd

    if not curated:
        if bout_pd.index.size > 0:
            bs.cleanup(bout_pd)
        # if it comes up nonempty after cleanup, get spectrograms, etc
        if bout_pd.index.size > 0:
            # if it is something:
            if hparams['sample_rate'] is None:
                one_wav_path = bout_pd.loc[0, 'file']
                logger.info(
                    'Sample rate not saved in parameters dict, searching it in ' + one_wav_path)
                hparams['sample_rate'] = bs.sample_rate_from_wav(one_wav_path)
            if not ('spectrogram' in bout_pd.keys()):
                logger.info('No spectrograms in here, will compute...')
                bout_pd['spectrogram'] = bout_pd['waveform'].apply(
                    lambda x: bs.gimmepower(x, hparams)[2])
                #logger.info('saving bout pandas with spectrogram to ' + bouts_auto_file_path)

            if not ('confusing' in bout_pd.keys()):
                bout_pd['confusing'] = True

            if not ('bout_check' in bout_pd.keys()):
                bout_pd['bout_check'] = False
            bout_pd.reset_index(drop=True, inplace=True)
            # check whether it has the spectrogram and compute it if it doesnt compute them
            # reset the index
            # add the curation tags
    return hparams, bout_pd


# for summary and so forth
default_bout_sess_par = {'bird': None,
                         'acq_software': 'alsa',
                         'derived_folder': 'bouts_ceciestunepipe',
                         'auto_file': 'bout_auto_file',
                         'curated_file': 'bout_curated_file',
                         'super_session': 'all-sess-01'}


def list_sessions(sess_par: dict) -> np.array:
    sess_arr = np.array(et.list_sessions(sess_par['bird'], section='raw',
                                         ephys_software=sess_par['acq_software']))
    sess_arr.sort()
    return sess_arr


def get_bird_sess_pd(sess_par: dict) -> pd.DataFrame:
    # go through all sessions with raw data and check which one has auto bouts, curated bouts.
    # get the sess_arr of raw, which ever raw is
    sess_arr = list_sessions(sess_par)
    sess_pd = pd.DataFrame({'sess': sess_arr})
    sess_pd['acq_soft'] = sess_par['acq_software']

    # see which has auto bouts
    sess_pd['has_auto_bouts'] = sess_pd['sess'].apply(lambda x: has_bouts_file(sess_par['bird'], x, sess_par['acq_software'],
                                                                               derived_folder=sess_par['derived_folder'],
                                                                               bout_type=sess_par['auto_file'])
                                                      )

    if 'curated_file' in sess_par.keys():
        sess_pd['has_curated_bouts'] = sess_pd['sess'].apply(lambda x: has_bouts_file(sess_par['bird'], x, sess_par['acq_software'],
                                                                                      derived_folder=sess_par['derived_folder'],
                                                                                      bout_type=sess_par['curated_file'])
                                                             )
    else:
        sess_pd['has_curated_bouts'] = None
    return sess_pd


def load_bouts_ds(s_ds: pd.Series, sess_par: dict,
                  file_key: str = 'auto_file', exclude_cols=[]) -> pd.DataFrame:
    # Load the bouts for a data series, exclude the cols, if any and return the loaded bout
    logger.debug('sess {}'.format(s_ds['sess']))
    hparams, b_pd = load_bouts(sess_par['bird'], s_ds['sess'], sess_par['acq_software'],
                               derived_folder=sess_par['derived_folder'],
                               bout_file_key=sess_par[file_key])

    # this should be fixed in sb.read_session_bouts
    b_pd = pd.DataFrame() if b_pd is None else b_pd
    if b_pd.index.size == 0:
        logger.warning('Bout pandas {} pickle was empty for sess {} parameters {}'.format(file_key,
                                                                                          s_ds['sess'],
                                                                                          sess_par))
    else:
        b_pd['sess'] = s_ds['sess']
        b_pd.drop(columns=exclude_cols, inplace=True)
        if sess_par['acq_software'] == 'alsa':
            # there is a bug in the pipeline and for bird s_b1555_22, in april 2022 some bout pd in alsa
            # contain pointers to the sglx files.
            # they can be identified by the wav_mic.wav file, instead of H-m-s-part.wav
            # if that is detected, warn and return empty dataframe. We'll have no timestamps for those yet.
            wav_files = np.unique(
                [os.path.split(x)[-1].split('.wav')[0] for x in b_pd['file']])
            if 'wav_mic' in wav_files:
                logger.warning('Bout pandas {} pickle is screwed up for sess {} parameters {}'.format(file_key,
                                                                                                      s_ds['sess'],
                                                                                                      sess_par))
                logger.warning(
                    'It seems to contain bouts from ephys wav files: {}'.format(wav_files))
                # for now, ignore. In the future, we will get the timestamp for those files
                b_pd = pd.DataFrame()
            else:
                b_pd = bs.alsa_bout_time_stamps(b_pd)
    return b_pd


def load_all_bouts(sess_par: dict,
                   exclude_cols: list = ['waveform', 'spectrogram', 'p_step'],
                   meta_pd: pd.DataFrame = None,
                   save: bool = True) -> tuple:
    # get the pandas with the sessions
    logger.info(
        'Looking for all sessions with bouts detected/curated for bird {}'.format(sess_par['bird']))

    if meta_pd is None:
        s_pd = get_bird_sess_pd(sess_par)
        sess_sel = True
    else:
        logger.info('Will only do sessions {}'.format(list(meta_pd['sess'])))
        s_pd = meta_pd
        sess_sel = True

    n_auto = np.sum((s_pd['has_auto_bouts'] == True) & (sess_sel))
    n_curated = np.sum((s_pd['has_curated_bouts'] == True) & (sess_sel))
    logger.info('Found {} sessions with detected, {} with curated bouts'.format(
        n_auto, n_curated))

    # load all curated, drop the
    sel_auto = (s_pd['has_auto_bouts']) & sess_sel
    auto_bout_pd_list = list(s_pd.loc[sel_auto].apply(lambda s: load_bouts_ds(s, sess_par,
                                                                              exclude_cols=exclude_cols,
                                                                              file_key='auto_file'),
                                                      axis=1))
    auto_bout_pd = pd.concat(auto_bout_pd_list).rename(
        columns={'bout_check': 'bout_auto'})

    # auto bout ready to merge with curated bouts, wherever they exist
    # load the curated, wherever they exist, and if it is within the session parameters plan
    if 'curated_file' in sess_par.keys():
        sel_curated = (s_pd['has_curated_bouts']) & (sess_sel)
        check_bout_pd_list = list(s_pd.loc[sel_curated].apply(lambda s: load_bouts_ds(s, sess_par,
                                                                                      exclude_cols=exclude_cols,
                                                                                      file_key='curated_file'),
                                                              axis=1))
        check_bout_pd = pd.concat(check_bout_pd_list)
        # do the merge now
        bout_pd = auto_bout_pd.merge(check_bout_pd[['t_stamp', 'is_call', 'confusing', 'bout_check']],
                                     on='t_stamp',
                                     how='outer')
    else:
        bout_pd = auto_bout_pd

    bout_pd['datetime'] = pd.to_datetime(bout_pd['t_stamp'])
    bout_pd['day'] = bout_pd['datetime'].apply(
        lambda dt: dt.strftime('%Y-%m-%d'))
    bout_pd['hour'] = bout_pd['datetime'].apply(lambda x: x.hour)

    # save it
    if save:
        save_bouts_summary(s_pd, bout_pd, sess_par['bird'],
                           sess=sess_par['super_session'],
                           acq_soft=sess_par['acq_software'],
                           derived_folder=sess_par['derived_folder'])

    return s_pd, bout_pd


def update_bouts(sess_par: dict, exclude_cols: list = ['waveform', 'spectrogram', 'p_step']) -> tuple:
    logger.info(
        'Looking for all sessions with bouts detected/curated for bird {}'.format(sess_par['bird']))

    # try loading. If there is anything, update it. If there is nothing, just do it
    try:
        # if there is anything, prep to update
        prev_meta_df, prev_bout_df = load_bouts_summary(sess_par['bird'],
                                                        sess=sess_par['super_session'],
                                                        acq_soft=sess_par['acq_software'],
                                                        derived_folder=sess_par['derived_folder'])
        #logger.info(prev_meta_df['sess'])

    except:
        # If nothing, just flag to load from scratch with prev_meta_df = None
        logger.warn(
            'Could not load meta/bouts files. Will just make everythin from scratch')
        prev_meta_df = None

    if prev_meta_df is None:
        # if flagged, load from scratch and that's it
        meta_df, bout_df = load_all_bouts(sess_par, exclude_cols=exclude_cols)

    else:
        # otherwise, update from the prev_meta_df, prev_bout_df
        new_meta_df = get_bird_sess_pd(sess_par)
        revisit_sess = np.unique(pd.concat([prev_meta_df, new_meta_df]).drop_duplicates(keep=False)['sess'])
        
        # if there is anything new update, otherwise just return what was loaded (no need to save)
        if revisit_sess.size > 0:
            logger.info('Will revisit sessions {}'.format(revisit_sess))
            # remove anything with those sessions from the bout_df, meta_df.
            #logger.info('pev bout before dropping {}'.format(np.unique(prev_bout_df['sess'])))
            prev_bout_df = prev_bout_df[~prev_bout_df['sess'].isin(revisit_sess)]
            prev_bout_df.reset_index(drop=True, inplace=True)
            #logger.info('pev bout after dropping {}'.format(np.unique(prev_bout_df['sess'])))

            # get meta_df, bout_df for just those sessions
            redo_meta_df = new_meta_df[new_meta_df['sess'].isin(revisit_sess)]
            logger.info('There are {} sessions to update'.format(redo_meta_df.index.size))
            _, redo_bout_df = load_all_bouts(sess_par,
                                             exclude_cols=exclude_cols,
                                             meta_pd=redo_meta_df,
                                             save=False)
            
            # concatenate
            meta_df = new_meta_df
            meta_df.reset_index(drop=True, inplace=True)
            bout_df = pd.concat([prev_bout_df, redo_bout_df])
            bout_df.reset_index(drop=True, inplace=True)
            #logger.info('concat bout {}'.format(np.unique(bout_df['sess'])))
            # save
            
            save_bouts_summary(meta_df, bout_df, sess_par['bird'],
                               sess=sess_par['super_session'],
                               acq_soft=sess_par['acq_software'],
                               derived_folder=sess_par['derived_folder'])
        
        else:
            logger.info('Nothing to update')
            meta_df = prev_meta_df
            bout_df = prev_bout_df

    # drop the sessions to update from the
    return meta_df, bout_df


def bout_summary_path(bird: str, sess: str = 'all-sess-01', acq_soft: str = 'alsa', derived_folder: str = 'bouts_ceciestunepipe') -> str:
    exp_struct = et.get_exp_struct(bird, sess, ephys_software=acq_soft)
    bout_file_path = os.path.join(
        exp_struct['folders']['processed'], derived_folder, 'bout_summary_df.pickle')
    meta_file_path = os.path.join(
        exp_struct['folders']['processed'], derived_folder, 'bout_meta_df.pickle')
    logger.info('Meta, bout summary path is {}, {}'.format(
        meta_file_path, bout_file_path))
    return meta_file_path, bout_file_path


def load_bouts_summary(bird: str, sess: str = 'all-sess-01', acq_soft: str = 'alsa', derived_folder: str = 'bouts_ceciestunepipe') -> pd.DataFrame:
    # file location
    logger.info('Loading bout summary dataframe')
    pickle_paths = bout_summary_path(
        bird, sess=sess, acq_soft=acq_soft, derived_folder=derived_folder)
    meta_df = pd.read_pickle(pickle_paths[0])
    bout_df = pd.read_pickle(pickle_paths[1])
    return meta_df, bout_df


def save_bouts_summary(meta_df: pd.DataFrame, bout_df: pd.DataFrame, bird: str,
                       sess='all-sess-01',
                       acq_soft='alsa',
                       derived_folder='bouts_ceciestunepipe') -> pd.DataFrame:
    # file location
    logger.info('Saving bout summary dataframe')
    pickle_paths = bout_summary_path(
        bird, sess=sess, acq_soft=acq_soft, derived_folder=derived_folder)
    fu.makedirs(os.path.split(pickle_paths[0])[0], exist_ok=True, mode=0o777)
    meta_df.to_pickle(pickle_paths[0])
    bout_df.to_pickle(pickle_paths[1])
    for p in pickle_paths:
        try:
            fu.chmod(p, 0o777)
        except PermissionError:
        # can't change permission because you don't own the file
            logger.warning('Cant change permission to file {} because you dont own it'.format(p))
    return pickle_paths


def plot_bout_stats(bout_pd: pd.DataFrame, zoom_days: int = 'all', bout_len_min: int=0, ax_dict: dict = None) -> dict:
    if ax_dict is None:
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(3, 3, figure=fig)
        ax_dict = { 'hourly': fig.add_subplot(gs[0, :2]),
                    'len': fig.add_subplot(gs[0, -1]),
                   'daily': fig.add_subplot(gs[1,:]),
                   'daily_len': fig.add_subplot(gs[2,:])}
        fig.suptitle('Bouts summary zoomed to last {} days and longer than {} ms'.format(zoom_days, bout_len_min))

    bout_pd['bout_check'].fillna(False, inplace=True)
    ### filter date to the last n days
    if zoom_days == 'all':
        from_date = bout_pd['datetime'].dt.date.min() - datetime.timedelta(days=1)
    else:
        from_date = bout_pd['datetime'].dt.date.max() - datetime.timedelta(days=zoom_days)

    date_filter = bout_pd['datetime'].dt.date > from_date
    len_filter = bout_pd['len_ms'] > bout_len_min
    confusing_filter = bout_pd['confusing']==False
    all_filter = date_filter & len_filter & confusing_filter
    bout_pd = bout_pd[all_filter]
      
    ax_h = ax_dict['hourly']
    bout_pd.groupby(bout_pd['datetime'].dt.hour)['bout_auto'].sum().plot(kind='bar', ax=ax_h, alpha=0.5)
    bout_pd.groupby(bout_pd['datetime'].dt.hour)['bout_check'].sum().plot(kind='bar', ax=ax_h, alpha=0.5, color='red')
    ax_h.set_xlabel('Hour')
    ax_h.set_ylabel('bouts')

    ax_l = ax_dict['len']
    bout_pd.groupby(bout_pd['len_ms']//1000)['bout_auto'].sum().plot(ax=ax_l, alpha=0.5)
    bout_pd.groupby(bout_pd['len_ms']//1000)['bout_check'].sum().plot(ax=ax_l, alpha=0.5, color='red')
    ax_l.set_xlabel('Length (seconds)')
    ax_l.set_ylabel('bouts')
    ax_l.set_xlim(0, np.max(bout_pd['len_ms']//1000)+5)
    
    ax_d = ax_dict['daily']
    bout_pd.groupby(bout_pd['datetime'].dt.date)['bout_auto'].sum().plot(kind='bar', ax=ax_d, alpha=0.5, label='auto')
    bout_pd.groupby(bout_pd['datetime'].dt.date)['bout_check'].sum().plot(kind='bar', ax=ax_d, alpha=0.5, 
    color='red', label='curated')
    ax_d.set_xlabel('Day')
    ax_d.set_ylabel('bouts')
    ax_d.legend()
    #ax_d.xaxis.set_ticklabels([])

    ax_dl = ax_dict['daily_len']
    # (bout_pd.groupby(bout_pd['datetime'].dt.date)['len_ms'].sum()/60000).plot(kind='bar', ax=ax_dl, alpha=0.5, 
    # label='auto')
    (bout_pd.loc[bout_pd['bout_check']==True, :].groupby(bout_pd.loc[bout_pd['bout_check']==True, 'datetime'].dt.date)['len_ms'].sum()/60000).plot(kind='bar', ax=ax_dl, alpha=0.5, 
    color='red', label='curated')
    ax_dl.set_xlabel('Day')
    ax_dl.set_ylabel('song len (min)')
    
    
    plt.tight_layout()
    return ax_dict


def main():
    """Launcher"""
    # make a logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('Running searchbout on {}'.format(socket.gethostname()))

    # get all birds
    days_lookup = 2
    do_today = False
    force_compute = False
    #birds_list = ['s_b1575_23', 's_b1515_23']
    birds_list = [] #empty list will run through all

    get_starlings_alsa_bouts(
        days_lookup, force=force_compute, do_today=do_today, birds_list=birds_list)

    # get_birds_bouts(['s_b1575_23', 's_b1515_23'], days_lookup, bs.default_hparams,
    # ephys_software='alsa',
    # n_jobs=12,
    # force=True,
    # do_today=do_today)

    #get_one_day_bouts('s_b1555_22', '2022-08-06')
    # apply filters if any

    logger.info('done for the day')
    return 0


if __name__ == "__main__":
    sys.exit(main())

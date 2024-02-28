# functions to go through spikeglx experiments using spikeinterface
# read metada, concatenate, align and sort
import os
import sys
import glob
import logging
import datetime
import shutil
import numpy as np
import pandas as pd
import warnings
import contextlib
import multiprocessing
import pickle

#from intan2kwik.core.h5 import tables

import spikeinterface as si
from spikeinterface import extractors as se
from spikeinterface import toolkit as st
from spikeinterface import sorters as ss
from spikeinterface import comparison as sc

from ceciestunepipe.file import filestructure as et
from ceciestunepipe.util import sglxutil as sglu
from ceciestunepipe.util import fileutil as fu

from ceciestunepipe.util.spikeextractors.extractors.spikeglxrecordingextractor import spikeglxrecordingextractor as sglex


logger = logging.getLogger('ceciestunepipe.util.sglexp')

N_JOBS_MAX = multiprocessing.cpu_count()

def make_sgl_epoch_dict(sess_par_dict: dict, epoch: str) -> pd.DataFrame:

    exp_struct = sglu.sgl_struct(sess_par_dict, epoch)
    sgl_folders, sgl_files = sglu.sgl_file_struct(exp_struct['folders']['raw'])

    streams_list = list(sgl_files.keys())

    # make the dataframe with the file locations
    epoch_pd = pd.DataFrame(sgl_files)
    n_runs = epoch_pd.index.size
    epoch_pd['run'] = range(n_runs)
    epoch_pd['sess'] = sess_par_dict['sess']
    epoch_pd['bird'] = sess_par_dict['bird']
    epoch_pd['epoch'] = epoch
    epoch_pd = epoch_pd.reindex(
        ['bird', 'sess', 'epoch', 'run'] + streams_list, axis=1)

    # get the recording for every meta file in each run
    for stream in streams_list:
        epoch_pd['rec-' + stream] = epoch_pd[stream].apply(
            lambda x: sglex.SpikeGLXRecordingExtractor(sglu.get_data_meta_path(x)[0]))

    return epoch_pd


def make_session_pd(sess_par_dict: dict, exclude_suffix='all') -> pd.DataFrame:

    sess_epochs = sglu.list_sgl_epochs(sess_par_dict)
    # do not count hand-merged sessions
    sess_epochs_list = [s for s in sess_epochs if not(exclude_suffix in s)]

    sess_name = sess_par_dict['sess']
    logger.info('session {} has epochs {}'.format(sess_name, sess_epochs_list))

    df_list = [make_sgl_epoch_dict(sess_par_dict, epoch)
               for epoch in sess_epochs_list]

    sess_pd = pd.concat(df_list, axis=0)
    sess_pd.reset_index(drop=True, inplace=True)

    return sess_pd


def merge_session_pd(sess_pd: pd.DataFrame) -> pd.DataFrame:

    # get the list of existing recordings (one per stream)
    rec_list = [c for c in (sess_pd.columns) if 'rec' in c]

    # merge each column
    merged_rec_dict = {c: se.MultiRecordingTimeExtractor(
        recordings=list(sess_pd.loc[:, c].values)) for c in rec_list}

    merged_pd = pd.DataFrame.from_dict(merged_rec_dict, orient='index').T
    merged_pd['bird'] = sess_pd['bird']
    merged_pd['sess'] = sess_pd['sess']
    merged_pd['epoch'] = [list(sess_pd.loc[:, 'epoch'].values)]*len(merged_pd)
    merged_pd['run'] = [list(sess_pd.loc[:, 'run'].values)]*len(merged_pd)

    return merged_pd


def run_spikesort(recording_extractor: se.RecordingExtractor,
                  logger: logging.Logger,
                  sort_pickle_path: str,
                  tmp_dir: str,
                  grouping_property: str = None,
                  sorting_method: str = 'kilosort2',
                  n_jobs_bin: int = N_JOBS_MAX,
                  chunk_mb: int = 4096, 
                  restrict_to_gpu=None,
                  **sort_kwargs):

    logger.info("Grouping property: {}".format(grouping_property))
    logger.info("sorting method: {}".format(sorting_method))

    # try:
    if sorting_method == "kilosort2":
        # perform kilosort sorting
        sort_tmp_dir = os.path.join(tmp_dir, 'tmp_ks2')
        logger.info('Sorting tmp dir {}'.format(sort_tmp_dir))

        if restrict_to_gpu is not None:
            logger.info(
                'Will set visible gpu devices {}'.format(restrict_to_gpu))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(restrict_to_gpu)

        sort = ss.run_kilosort2(
            recording_extractor,
            car=True,
            output_folder=sort_tmp_dir,
            parallel=True,
            verbose=True,
            grouping_property=grouping_property,
            chunk_mb=chunk_mb,
            n_jobs_bin=n_jobs_bin,
            **sort_kwargs
        )

    else:
        raise NotImplementedError('Only know how to sort kilosort2 for now, \
                                        will deal with {} later'.format(sorting_method))

    logger.info('done sorting')

    # save sort
    logger.info("Saving sort {}".format(sort_pickle_path))
    with open(sort_pickle_path, "wb") as output:
        pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
    logger.info("Sorting output saved to {}".format(sort_pickle_path))

    # get templates and max channel
    logger.info("Getting templates")
    templates = st.postprocessing.get_unit_templates(
        recording_extractor,
        sort,
        max_spikes_per_unit=200,
        save_as_property=True,
        verbose=True,
        n_jobs=n_jobs_bin,
        grouping_property=grouping_property,
    )

    logger.info("Getting main channel")
    max_chan = st.postprocessing.get_unit_max_channels(
        recording_extractor,
        sort,
        save_as_property=True,
        verbose=True,
        n_jobs=n_jobs_bin
    )

    # save sort again with all that processed data
    pickle_folder, pickle_filename = os.path.split(sort_pickle_path)
    temp_pickle_filename = pickle_filename.split('.pickle')[0] + 'temp.pickle'
    sort_temp_pickle_path = os.path.join(pickle_folder, temp_pickle_filename)
    logger.info("Saving sort {}".format(sort_temp_pickle_path))
    sort.dump_to_pickle(sort_temp_pickle_path)

    return sort

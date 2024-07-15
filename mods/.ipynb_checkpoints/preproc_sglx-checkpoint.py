import os
import json
import logging
import traceback
import warnings
import socket
from ceciestunepipe.file import bcistructure as et
from ceciestunepipe.util.spikeextractors import preprocess as pre

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('Running on {}'.format(socket.gethostname()))


### sequentially process all runs of the sessions
def preprocess_session(sess_par: dict,force_redo=False,skip_wav=True):
    logger.info('pre-process all runs of sess ' + sess_par['sess'])
    # get exp struct
    sess_struct = et.get_exp_struct(sess_par['bird'],sess_par['sess'],sort=sess_par['sort'])
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
        sess_epochs = et.list_sgl_epochs(sess_par)
        logger.info('found epochs: {}'.format(sess_epochs))
        # preprocess all epochs
        epoch_dict_list = []
        for i_ep, epoch in enumerate(sess_epochs):
            try:
                exp_struct = et.sgl_struct(sess_par,epoch)
                one_epoch_dict = pre.preprocess_run(sess_par,exp_struct,epoch,skip_wav=skip_wav)
                epoch_dict_list.append(one_epoch_dict)
            except Exception as exc:
                warnings.warn('Error in epoch {}'.format(epoch), UserWarning)
                logger.info(traceback.format_exc)
                logger.info(exc)
                logger.info('Session {} epoch {} could not be preprocessed'.format(sess_par['sess'], epoch))
                

##### zeke sync load stim                
def load_stim_tags_dict(bout_stim_sess: str,bird_str: str) -> dict:
    sess_exp_struct = et.get_exp_struct(bird_str, bout_stim_sess)
    sess_derived_folder = os.path.split(sess_exp_struct['folders']['derived'])[0]
    stim_tags_path = os.path.join(sess_derived_folder, 'bout_stim', 'sbc_stim', 'stim_tags.json')
    with open(stim_tags_path, 'rb') as f:
        stim_tags_dict = json.load(f)
    return stim_tags_dict
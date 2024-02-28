import numpy as np
import pandas as pd
import logging
import os
import pickle

from scipy.io import wavfile

from ceciestunepipe.file import bcistructure as et

from ceciestunepipe.util import sglxutil as sglu
from ceciestunepipe.util import rigutil as ru
from ceciestunepipe.util import wavutil as wu
from ceciestunepipe.util import syncutil as su
from ceciestunepipe.util import stimutil as st
from ceciestunepipe.util import fileutil as fu
from ceciestunepipe.util.spikeextractors.extractors.spikeglxrecordingextractor import spikeglxrecordingextractor as sglex
import spikeinterface.extractors as se

logger = logging.getLogger('ceciestunepipe.util.spikeextractors.preprocess')

def load_sglx_recordings(exp_struct: dict, epoch:str) -> tuple:
    # get the folders
    # make a pandas of the files
    # read into spikextractors
    # get the rig parameters
    # extract the wav channels of the nidaq signals (mics, syn)

    sgl_folders, sgl_files = sglu.sgl_file_struct(exp_struct['folders']['sglx'])
    logger.info('loading files {}'.format(sgl_files))
    files_pd = pd.DataFrame(sgl_files)

    i_run = 0 #spikeglx epochs have the one run, this is for later
    run_meta_files = {k: v[i_run] for k, v in sgl_files.items()}
    run_recordings = {k: sglex.SpikeGLXRecordingExtractor(sglu.get_data_meta_path(v)[0]) for k, v in run_meta_files.items()}

    rig_dict = ru.get_rig_par(exp_struct)

    logger.info('Got sglx recordings for keys {}'.format(list(run_recordings.keys())))

    return run_recordings, run_meta_files, files_pd, rig_dict 

def extract_nidq_channels(sess_par, run_recs_dict, rig_dict, chan_name_list, chan_type='adc') -> np.array:
    # get the channels id, numbers for the mic_list
    chan_n_list = [int(ru.lookup_signal(rig_dict, n)[1].split('-')[-1]) for n in chan_name_list]
    
    if chan_type=='adc':
        stream = run_recs_dict['nidq'].get_traces(channel_ids=chan_n_list)
    elif chan_type=='ttl':
        stream = run_recs_dict['nidq'].get_ttl_traces()[chan_n_list, :]
    else:
        raise NotImplementedError('dont know how to deal with {} channels'.format(chan_type))
        
    return stream

def save_wav(stream: np.array, s_f: float, wav_path: str) -> int:
    
    wav_s_f = int(round(s_f/1000))*1000
    logger.info('sampling rate {}'.format(wav_s_f))
    
    # write to wav
    logger.info('saving {}-shaped array as wav in {}'.format(stream.shape, wav_path))
    os.makedirs(os.path.split(wav_path)[0], exist_ok=True, mode=0o777)
    wavfile.write(wav_path, wav_s_f, stream.T)
    
    ## also save as numpy
    npy_path = wav_path.split('.')[0] + '.npy'
    logger.info('saving {}-shaped array as npy in {}'.format(stream.shape, npy_path))
    np.save(npy_path, stream.T)
    return wav_s_f

def chans_to_wav(recording_extractor, chan_list: list, wav_path:str, only_npy: bool=False) -> int:
    # get the stream
    data_stream = recording_extractor.get_traces(channel_ids=chan_list)
    # make sure folder exists
    logger.info('saving {}-shaped array as wav in {}'.format(data_stream.shape, wav_path))
    os.makedirs(os.path.split(wav_path)[0], exist_ok=True, mode=0o777)
    # write it
    s_f = int(round(recording_extractor.get_sampling_frequency()/1000))*1000
    logger.info('sampling rate {}'.format(s_f))
    wavfile.write(wav_path, s_f, data_stream.T)
    
    ## also save as numpy
    npy_path = wav_path.split('.')[0] + '.npy'
    logger.info('saving {}-shaped array as npy in {}'.format(data_stream.shape, npy_path))
    np.save(npy_path, data_stream.T)
    return s_f

def ttl_signal_to_npy(recording_extractor, rig_dict, signal_name, npy_path: str) -> np.array:
    # get a channel order by its name
    # extract the digital events
    # save in a numpy array (n_events, 2) with (stamp, event) in each column
    logger.info('looking for signal {} in the nidaq channels'.format(signal_name))
    chan_id = int(ru.lookup_signal(rig_dict, signal_name)[1].split('-')[-1])
    logger.info('found in chan {}'.format(chan_id))

    ttl_tuple = recording_extractor.get_ttl_events(channel_id=chan_id)
    ttl_arr = np.vstack(ttl_tuple).T
    
    logger.info('saving {}-shaped array as npy in {}'.format(ttl_arr.shape, npy_path))
    np.save(npy_path, ttl_arr)
    return ttl_tuple

def load_syn_dict(exp_struct: dict, stream:str, arrays_to_load=['evt_arr', 't_0']) -> dict:
    
    syn_dict_path = os.path.join(exp_struct['folders']['derived'],  
                                 '{}_sync_dict.pkl'.format(stream))
    logger.info('loading syn_dict from ' + syn_dict_path)
    with open(syn_dict_path, 'rb') as f:
        syn_dict = pickle.load(f)
     
    for arr in arrays_to_load:
        arr_path = syn_dict['{}_path'.format(arr)]
        syn_dict[arr] = np.load(arr_path, mmap_mode='r')
    return syn_dict

def get_syn_pattern(run_recs_dict, exp_struct, stream:str, force=False):
    logger.info('getting syn patterns for {}'.format(stream))
    syn_dict_path = os.path.join(exp_struct['folders']['derived'], '{}_sync_dict.pkl'.format(stream))
    
    # if Force=true of file not found, compute it from the recording_dict
    if not(os.path.exists(syn_dict_path) and (force is False)):
        logger.info('File {} not found or forced computation, getting the events'.format(syn_dict_path))
        if stream=='wav':
            raise NotImplementedError('Dont know how to force extraction of syn signals from wav channels here. Go back to preprocessing')
        
        ## get syn from the imec channels
        syn_tuple, syn_arr = get_syn_imec(run_recs_dict[stream])
        
        t_0_path = os.path.join(exp_struct['folders']['derived'],  '{}_t0.npy'.format(stream))
        syn_npy_path = os.path.join(exp_struct['folders']['derived'],  '{}_sync_evt.npy'.format(stream))
                                 
        logger.info('saving events array to ' + syn_npy_path)
        np.save(syn_npy_path, syn_arr)
        
        logger.info('saving t_0 array to ' + t_0_path)                                 
        np.save(t_0_path, syn_tuple[1])
        
        syn_dict = {'path': syn_dict_path,
                    's_f': syn_tuple[0],
                   't_0_path': t_0_path,
                    'evt_arr_path': syn_npy_path
                   }
        
        # save without the array, and open the array as a memmap
        logger.info('saving sync dict to ' + syn_dict_path)
        with open(syn_dict_path, 'wb') as pf:
            pickle.dump(syn_dict, pf)

    ## in any case, load the saved dict so everything comes from the memmaped arrays
    syn_dict = load_syn_dict(exp_struct, stream) 
    return syn_dict

def get_syn_imec(run_sglx_recording: sglex.SpikeGLXRecordingExtractor) -> tuple:
    ## get syn from the imec channels
    syn_tuple = run_sglx_recording.get_effective_sf(force_ttl=True)
    syn_arr = np.vstack(list(syn_tuple[2][:]))
    return syn_tuple, syn_arr



def sync_all(all_syn_dict: dict, ref_stream: str, force=False) -> dict:
    logger.info('syncing all times to {}'.format(ref_stream))
    ref_syn_dict = all_syn_dict[ref_stream]
    for one_stream, one_syn_dict in all_syn_dict.items():
        if one_stream==ref_stream:
            continue
            
        logger.info(' synch {}...'.format(one_stream))
        
        t_0_folder = os.path.split(one_syn_dict['t_0_path'])[0]
        t_p_path = os.path.join(t_0_folder, '{}-tp.npy'.format(one_stream))
        
        if not(os.path.exists(t_p_path) and (force is False)):
            logger.info('  t_prime file {} not found or forced computation, getting the events'.format(t_p_path))
        
            t_prime = su.sync_to_pattern(one_syn_dict['evt_arr'], one_syn_dict['t_0'],
                                                     ref_syn_dict['evt_arr'], ref_syn_dict['t_0'])
            logger.info('    saving t_prime array to ' + t_p_path)                                 
            np.save(t_p_path, t_prime)
        
            #clar the memory, then load as memmap
            del t_prime
        
            one_syn_dict['t_p_path'] = t_p_path
            # save the dict with the path to the sync it
            logger.info('    saving synced dict to {}'.format(one_syn_dict['path']))
            with open(one_syn_dict['path'], 'wb') as fp:
                pickle.dump(one_syn_dict, fp)
            
        one_syn_dict['t_p'] = np.load(t_p_path, mmap_mode='r')
    return

def preprocess_run(sess_par: dict, exp_struct: dict, epoch:str, do_sync_to_stream=None, skip_wav=False) -> dict:
    # get the recordings
    logger.info('PREPROCESSING sess {} | epoch {}'.format(sess_par['sess'], epoch))
    logger.info('getting extractors')
    sgl_exp_struct = et.sgl_struct(sess_par, epoch)
    run_recs_dict, run_meta_files, files_pd, rig_dict = load_sglx_recordings(sgl_exp_struct, epoch)
    
    # go through the sglx files of the session extracting data and events, according to the metadata found 
    # in the rig.json file

    ### For all the nidaq channels:
    # get the sampling rate
    nidq_s_f = run_recs_dict['nidq'].get_sampling_frequency()
    
    ## get the microphone(s) to wav
    # get the chans
    mic_list = sess_par['mic_list']
    logger.info('Getting microphone channel(s) {}'.format(mic_list))
    # if multiple mics, will save as multiple channels within wav file
    mic_stream = extract_nidq_channels(sess_par, run_recs_dict, rig_dict, mic_list, chan_type='adc')
    mic_file_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_mic.wav')
    wav_s_f = wu.save_wav(mic_stream, nidq_s_f, mic_file_path, skip_wav=skip_wav)

    ### if there were other adc channels (stim, for instance)
    ### get the stimulus signals to wav
    if 'adc_list' in sess_par:
        adc_list = sess_par['adc_list']
    else:
        adc_list = []
    if len(adc_list) > 0:
        adc_list = sess_par['adc_list']
        logger.info('Getting adc channel(s) {}'.format(adc_list))
        adc_stream = extract_nidq_channels(sess_par, run_recs_dict, rig_dict, adc_list, chan_type='adc')
        adc_file_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_adc.wav')
        wav_s_f = wu.save_wav(adc_stream, nidq_s_f, adc_file_path, skip_wav=skip_wav)
        adc_file_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_adc.npy')
        np.save(adc_file_path, adc_stream)

    if 'stim_list' in sess_par:
        stim_list = sess_par['stim_list']
    else:
        stim_list = []
    if len(stim_list) > 0:
        logger.info('Getting stimulus channel(s) {}'.format(stim_list))

        stim_stream = extract_nidq_channels(sess_par, run_recs_dict, rig_dict, stim_list, chan_type='adc')
        stim_file_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_stim.wav')
        wav_s_f = wu.save_wav(stim_stream, nidq_s_f, stim_file_path, skip_wav=skip_wav)

        if 'wav_syn' in stim_list:
            wav_syn_ch_in_stream = np.where(np.array(sess_par['stim_list']) == 'wav_syn')[0][0]
            
            logger.info('Getting the onset/offset of stimuli from the {} extracted analog channel'.format(wav_syn_ch_in_stream))
            wav_sync_stream = stim_stream[wav_syn_ch_in_stream]
            sine_ev, sine_ttl, t_ttl = st.get_events_from_sine_sync(wav_sync_stream, nidq_s_f, step_ms=100)

            sine_ttl_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_stim_sync_sine_ttl.npy')
            sine_ttl_t_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_stim_sync_sine_ttl_t.npy')
            sine_ev_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_stim_sync_sine_ttl_evt.npy')
            np.save(sine_ttl_path, sine_ttl)
            np.save(sine_ttl_t_path, t_ttl)
            np.save(sine_ev_path, sine_ev)
            logger.info('saved onset/offset of trial events from the sine wave in ' + sine_ev_path)

    # get the syn (from whatever TTL it was in) to wav
    sync_list = ['sync']
    logger.info('Getting sync channel(s) from nidaq streams: {}'.format(sync_list))
    sync_stream = extract_nidq_channels(sess_par, run_recs_dict, rig_dict, sync_list, chan_type='ttl')  
    sync_file_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_sync.wav')
    wav_s_f = wu.save_wav(sync_stream, nidq_s_f, sync_file_path, skip_wav=skip_wav)
    
    logger.info('Getting sync events from the wav sync channel')
    sync_ev_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_sync_evt.npy')
    wav_s_f, x_d, ttl_arr = wu.wav_to_syn(sync_file_path)
    logger.info('saving sync events of the wav channel to {}'.format(sync_ev_path))
    np.save(sync_ev_path, ttl_arr)
    
    t_0_path = os.path.join(sgl_exp_struct['folders']['derived'], 'wav_t0.npy')
    logger.info('saving t0 for wav channel to {}'.format(t_0_path))
    np.save(t_0_path, np.arange(sync_stream.size)/wav_s_f)
    
    # Get other digital channels
    if 'nidq_ttl_list' in sess_par:
        stim_list = sess_par['nidq_ttl_list']
    else:
        stim_list = []
    if len(stim_list) > 0:
        logger.info('Will get {} ttl signals'.format(len(stim_list)))
    
    for signal_name in stim_list:
        sig_npy_path = os.path.join(sgl_exp_struct['folders']['derived'], '{}_evt.npy'.format(signal_name))
        ttl_signal_to_npy(run_recs_dict['nidq'], rig_dict, signal_name, sig_npy_path)

    #make the sync dict
    nidq_syn_dict = {'s_f': wav_s_f,
           't_0_path': t_0_path,
           'evt_arr_path': sync_ev_path}
    
    nidq_syn_dict_path = os.path.join(sgl_exp_struct['folders']['derived'],  '{}_sync_dict.pkl'.format('wav'))
    nidq_syn_dict['path'] = nidq_syn_dict_path
    logger.info('saving sync nidaq dict to ' + nidq_syn_dict_path)
    with open(nidq_syn_dict_path, 'wb') as pf:
        pickle.dump(nidq_syn_dict, pf)

    ### get the sync for all other imec streams
    all_streams_list = list(run_recs_dict.keys())
    imec_streams = [x for x in all_streams_list if any(y in x for y in ['lf', 'ap'])]
    logger.info('Getting sync signals for imec streams: {}'.format(imec_streams))
    all_syn_dict = {k: get_syn_pattern(run_recs_dict, sgl_exp_struct, k, force=True) for k in imec_streams}
    
    # load all the nidq, wav syn_pattern made above, in the kosher way
    all_syn_dict['nidq'] = get_syn_pattern(run_recs_dict, sgl_exp_struct, 'nidq', force=False)
    all_syn_dict['wav'] = get_syn_pattern(run_recs_dict, sgl_exp_struct, 'wav', force=False)

    ### sync all to one pattern
    if do_sync_to_stream is not None:
        logger.info('Will do the sync to stream {}'.format(do_sync_to_stream))
        sync_all(all_syn_dict, do_sync_to_stream, force=True)

    # make the overall epoch pre-process dictt
    epoch_dict = {'epoch': epoch,
                'sgl_exp_struct': sgl_exp_struct,
                'files_pd': files_pd,
                 'recordings': run_recs_dict,
                 'meta': run_meta_files,
                 'rig': rig_dict,
                 'syn': all_syn_dict,
                 'sess_par': sess_par
                 }
    
    epoch_dict_path = os.path.join(sgl_exp_struct['folders']['derived'], 'preprocess_par.pickle')
    save_keys_list = ['epoch', 'sgl_exp_struct', 'files_pd', 'meta', 'rig', 'sess_par']
    epoch_dict_save = {k: epoch_dict[k] for k in save_keys_list}
    
    fu.save_pickle(epoch_dict_save, epoch_dict_path)
    logger.info('saved epoch preprocessing parameters to ' + epoch_dict_path)
    return epoch_dict
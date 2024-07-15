import os
import socket
import json
import logging

logger = logging.getLogger('ceciestunepipe.file.filestructure')

locations_dict = dict()
locations_dict['zebra'] = {'mnt': os.path.abspath('/mnt/zuperfinch/microdrive/birds'),
                           'local': os.path.abspath('/data/experiment/microdrive'),
                           'fast': os.path.abspath('/mnt/scratch/experiment')}

locations_dict['zpikezorter'] = {'mnt': os.path.abspath('/mnt/microdrive/birds'),
                                 'local': os.path.abspath('/data/experiment/microdrive')}

locations_dict['ZOROASTRO'] = {'mnt': os.path.abspath('B:\microdrive\data'),
                               'local': os.path.abspath('D:\microdrive')}

locations_dict['lookfar'] = {'mnt': os.path.abspath('/Volumes/Samsung_X5/microdrive'),
                             'local': os.path.abspath('/Volumes/Samsung_X5/microdrive'),
                             'fast': os.path.abspath('/Volumes/Samsung_X5/scratch')
                             }

# locations_dict['passaro'] = {'mnt': os.path.abspath('/mnt/cube/earneodo/basalganglia/birds'),
#                              'local': os.path.abspath('/Data/basalganglia'),
#                              'fast': os.path.abspath('/Data/scratch')
#                              }

locations_dict['passaro']= {'mnt': os.path.abspath('/mnt/cube/earneodo/basalganglia/birds'),
                             'local': os.path.abspath('/Data/raw_data'),
                             'fast': os.path.abspath('/mnt/cube/earneodo/scratch')
                             }

locations_dict['txori']= {'mnt': os.path.abspath('/mnt/sphere/speech_bci/'),
                             'local': os.path.abspath('/scratch/earneodo'),
                             'fast': os.path.abspath('/scratch/earneodo')
                             }


locations_dict['pakhi']= {'mnt': os.path.abspath('/mnt/cube/earneodo/bci_zf/neuropix/birds'),
                             'local': os.path.abspath('/mnt/sphere/earneodo/bci_zf/ss_data'),
                             'fast': os.path.abspath('/scratch/earneodo/tmp')
                             }

locations_dict['pouli']= {'mnt': os.path.abspath('/mnt/cube/earneodo/bci_zf/neuropix/birds'),
                             'local': os.path.abspath('/experiment/ss_data'),
                             'fast': os.path.abspath('/experiment/tmp')
                             }


# locations_dict['lookfar'] = {'mnt': os.path.abspath('/Users/zeke/experiment/birds'),
#                              'local': os.path.abspath('/Users/zeke/experiment/birds'),
#                              'fast': os.path.abspath('/Users/zeke/experiment/scratch')
#                              }
# Zinch in windows
locations_dict['Zinch'] = {'mnt': '\\\\ZUPERFINCHJR\storage\Data',
                           'local': os.path.abspath('C:\experiment')}

# Zinch in linux
locations_dict['zinch'] = {'mnt': '/mnt/zuperfinchjr/Data',
                           'local': os.path.abspath('/media/zinch/Windows/experiment')}

default_struct_par = {'neural': 'ephys',
                      'presort': 'kwik',
                      'sort': 'msort'}
# Todo: define locations_dict in a json file or something more elegant


def get_locations_from_hostname():
    hostname = socket.gethostname().split('.')[0]
    return locations_dict[hostname]


def read_json_exp_struct():
    raise NotImplementedError


def get_file_structure(location: dict, sess_par: dict) -> dict:
    """[summary]
    Arguments:
        location {dict} -- [description]
        sess_par {dict} -- session parameters dictionary. Example:
            sess_par = {'bird': 'p1j1',
            'sess': '2019-02-27_1800_02', 
            'probe': 'probe_0',
            'sort': 0} 
            - bird and sess are self-explanatory and refer to the folder of the raw, kwd files.
            - probe describes the probe that was used to do the sorting, which in turn determines
              neural port (via the rig.json file) and probe mapping (via rig.json file and 
              pipefinch.pipeline.filestructure.probes)
            - sort determines the version of sorting, in case multiple sorts were made on the same
              session (e.g different time spans, different configurations of the sorting algorithm)
              if the field is not present or is None, the .kwik, unit_waveforms and rasters will be directly
              in the Ephys\kwik\sess_id folder.
              otherwise, a 'sort_x' folder will contain the sorting files.

    Returns:
        dict -- ditcionary containing paths of folders and files.
            exp_struct['folders']: dictionary with ['raw', 'kwik', 'msort'] keys
            exp_struct['files']: dictionary with keys:
                'par': expermient.json path
                'set': settings.isf intan generated file path
                'rig': rig.json file desribing the recording settings (channels/signals)

                'kwd': kwd file with all raw data from the session
                'kwik': kwik file with sorted spikes
                'kwe': 
    """

    ephys_folder = 'Ephys'
    exp_struct = {}
    bird, sess = sess_par['bird'], sess_par['sess']

    exp_struct['folders'] = {}
    exp_struct['files'] = {}

    # The bird structure
    exp_struct['folders']['bird'] = os.path.join(location['mnt'], bird)
    
    # The raw files
    exp_struct['folders']['raw'] = os.path.join(
        location['mnt'], bird, ephys_folder, 'raw', sess)
    for f, n in zip(['par', 'set', 'rig'],
                    ['experiment.json', 'settings.isf', 'rig.json']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['raw'], n)

    # the kwik system (spikes, events, kwd file with streams)
    exp_struct['folders']['kwik'] = os.path.join(
        location['local'], bird, ephys_folder, 'kwik', sess)
    for f, n in zip(['kwd', 'kwik', 'kwe'], ['stream.kwd', 'spikes.kwik', 'events.kwe']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['kwik'], n)

    if 'sort' in sess_par and sess_par['sort'] is not None:
        exp_struct['files']['kwik'] = os.path.join(exp_struct['folders']['kwik'],
                                                   'sort_{}'.format(sess_par['sort']), 'spikes.kwik')

    # the processed system (dat_mic.mat, dat_ap.mat et. al files)
    exp_struct['folders']['processed'] = os.path.join(
        location['mnt'], bird, ephys_folder, 'processed', sess)
    for f, n in zip(['dat_mic', 'dat_ap', 'allevents'],
                    ['dat_mic.mat', 'dat_ap.mat', 'dat_all.pkl']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['processed'], n)

    # the 'derived' system (wav_mic, ...)
    exp_struct['folders']['derived'] = os.path.join(
        location['mnt'], bird, ephys_folder, 'derived', sess)
    for f, n in zip(['wav_mic'], ['wav_mic.wav']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['derived'], n)

    # the aux, temporary mountainsort files. these will be deleted after sorting
    # try 'fast' location first, if it does not exist, go for 'local'
    try:
        exp_struct['folders']['tmp'] = os.path.join(location['fast'], 'tmp')
    except KeyError:
        exp_struct['folders']['tmp'] = os.path.join(location['local'], 'tmp')
    
     # SET THE TMP DIRECTORY ENVIRONMENT VARIABLEM
    os.environ["TMPDIR"] = exp_struct['folders']['tmp']
    os.environ["TMP"] = exp_struct['folders']['tmp']

    try:
        msort_location = location['fast']
    except KeyError:
        msort_location = location['local']

    # MOUNTAINSORT FILE STRUCTURE
    exp_struct['folders']['msort'] = os.path.join(
        msort_location, bird, ephys_folder, 'msort', sess)
    for f, n in zip(['mda_raw', 'par'], ['raw.mda', 'params.json']):
        exp_struct['files'][f] = os.path.join(
            exp_struct['folders']['msort'], n)

    # KILOSORT FILE STRUCTURE
    exp_struct['folders']['ksort'] = os.path.join(
        msort_location, bird, ephys_folder, 'ksort', sess)
    for f, n in zip(['bin_raw', 'par'], ['raw.bin', 'params.json']):
        exp_struct['files'][f] = os.path.join(
            exp_struct['folders']['ksort'], n)

   

    return exp_struct


def get_exp_struct(bird, sess, sort=None, location_dict: dict = dict()):
    # get the configuration of the experiment:
    # if environment variable 'EXPERIMENT_PATH' exists,
    # read 'EXPERIMENT_PATH/config/expstruct.json'
    # no location dict was entered, try to get it from the hostname (from locations_dict)
    if location_dict:
        pass
    else:
        read_exp_base = os.environ.get('EXPERIMENT_PATH')

        if read_exp_base is not None:
            # if there is a .json file configured with the variables of the experiment
            exp_base = os.path.abspath(read_exp_base)
            location_dict_json_path = os.path.join(exp_base, 'exp_struct.json')
            location_dict = read_json_exp_struct()
        else:
            # try to read it from the hostname
            location_dict = get_locations_from_hostname()

    # make the exp struct dict.
    sess_par_dict = {'bird': bird,
                     'sess': sess,
                     'sort': sort}
    exp_struct = get_file_structure(location_dict, sess_par_dict)

    return exp_struct


def get_rig_par(exp_struct: dict) -> dict:
    rig_par_file = exp_struct['files']['rig']
    with open(rig_par_file, 'r') as fp:
        rig_par = json.load(fp)
    return rig_par

def make_all_events(exp_struct: dict) -> dict:
    mic_dat_path = exp_struct['files']['dat_mic']
    

def get_probe_port(exp_struct: dict, selected_probe: str) -> str:
    # get the probe and the port where the probe was connected
    rig_par = get_rig_par(exp_struct)
    probe_port = rig_par['chan']['port'][selected_probe].strip('-')
    return probe_port

def list_sessions(bird: str, location_dict: dict = dict(), section='raw') -> list:
    exp_struct = get_exp_struct(bird, '', location_dict=location_dict)
    return next(os.walk(exp_struct['folders'][section]))[1]

def msort_cleanup(exp_struct: dict):
    # remove the mda files
    mda_raw_path = exp_struct['files']['mda_raw']
    logger.info('removing (if exists) msort mda file {} '.format(mda_raw_path))
    try:
        os.remove(mda_raw_path)
    except FileNotFoundError:
        logger.debug('Nuttin done, file wasnt there')

def list_subfolders(folder_path):
    return next(os.walk(os.path.abspath(folder_path)))[1]
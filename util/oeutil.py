import os
import json
import pandas as pd
import glob

# this should go to eciestunepipe.file.bcistructure (as et here)
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



### tools for reading directly from oe
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
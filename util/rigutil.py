# tools to read/write the metadata of the rig.
# mainly, about the rig.json file
import json
import logging
import os

logger = logging.getLogger('ceciestunepipe.util.rigutil')
'''
example of a rig_par dictionary:
rig_par = {'chan': {'ttl': {'trig_perceptron': 'DIN-01', 'trig_recording': 'DIN-00'}, 
                   'adc': {'microphone_0': 'ADC-00'},
                   'port': {'probe_0': 'A-'}
                 },
           'probe': {'probe_0': {'model': 'a1x32-edge-5mm-20-177_H32',
                                'serial': '768b',
                                'headstage': 'intan32-h32'}}
          }
'''
def get_rig_par(exp_struct: dict) -> dict:
    # if there is a file for the run, get that
    # otherwise, assume the run gets same rig pars as the whole session
    try:
        with open(os.path.join(exp_struct['folders']['sglx'], 'rig.json')) as j_file:
            rig_dict = json.load(j_file)
    except FileNotFoundError:
        logger.debug('rig.json file not found for the run, going for the one for the session')
        with open(exp_struct['files']['rig']) as j_file:
            rig_dict = json.load(j_file)           
    return rig_dict

def lookup_signal(rig_par_dict: dict, signal_name: str) -> tuple:
    chan_dict = rig_par_dict['chan']
    for ch_type, chans in chan_dict.items():
        found = [channel for signal, channel in chans.items() if signal == signal_name]
        if len(found) > 0:
            found = found[0]
            break
        else:
            ch_type = None
            found = ''
    return ch_type, found

def get_probe_port(exp_struct:dict, selected_probe:str) -> str:
    # get the probe and the port where the probe was connected
    rig_par = get_rig_par(exp_struct)
    probe_port = rig_par['chan']['port'][selected_probe].strip('-')
    return probe_port
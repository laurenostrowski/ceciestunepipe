# Functions to do stuff with h5 open files
import logging
import h5py
import pandas as pd
import numpy as np

logger = logging.getLogger('ceciestunepipe.util.h5util')

bout_default_keys = {'attr_keys': ['start_ms', 'end_ms', 'len_ms',
                                   'start_sample', 'end_sample',
                                   'start_ms_ap_0', 'start_sample_ap_0',
                                   'start_sample_naive', 'len_sample_ap_0', 'end_sample_ap_0',
                                   'bout_check', 'confusing', 'file'],
                     'dset_keys': ['mic_arr', 'pre_arr', 'pre_filt', 'syl_in',
                                   'f_xx', 't_xx', 's_xx', 'sxx_pc',
                                   'spk_arr', 'spk_gpf'],
                     'dict_keys': ['s_f', 's_f_nidq', 's_f_ap_0', 't_pre', 't_post']
                     }


def h5_decorator(leave_open=True, default_mode='r'):
    """
    Decorator to open h5 structure if the path was provided to a function.
    :param h5_function: a function that receives an h5file as first argument
    :param leave_open: whether to leave the file open.
        It is overriden when file is entered open
    :param default_mode: what mode to open the file by default.
        It is overriden when file is entered open and when option 'mode' is set
        in h5_function (if it exists)
    :return: decorated function that takes open or path as first argument
    """
    def wrap(h5_function):
        def file_checker(h5_file, *args, **kwargs):
            if 'mode' in kwargs.keys():
                mode = kwargs['mode']
            else:
                mode = default_mode
            # logger.debug('leave open {}'.format(leave_open))
            logger.debug('mode {}'.format(mode))
            try:
                if type(h5_file) is not h5py._hl.files.File:
                    if leave_open:
                        logging.debug('Opening H5 file: {}'.format(h5_file))
                        h5_file = h5py.File(h5_file, mode)
                        return_value = h5_function(h5_file, *args, **kwargs)
                    else:
                        with h5py.File(h5_file, mode) as h5_file:
                            return_value = h5_function(h5_file, *args, **kwargs)

                else:
                    return_value = h5_function(h5_file, *args, **kwargs)
                return return_value

            except UnboundLocalError as err:
                last_err = err
                logger.error(err)

        return file_checker
    return wrap


def list_subgroups(h5_group):
    """
    List the groups within a group, not recursively
    :param h5_group: a Group object
    :return: a list of the keys of items that are instances of group
    """
    return [key for key, val in h5_group.items() if isinstance(val, h5py.Group)]


def dict_2_attr_translator(item):
    """
    Translate a value from a dictionary to a valid value for an attribute
    :param item: value of a read item of a dict (not a dict)
    :return: value to store as attribute
    """
    if item is None:
        value = 'None'
    else:
        value = item

    return value


def dict_2_group(parent_group, dic, name, replace=False):
    """
    Recursively dumps a dictionary into a group.
    keys go to attributes, values go to attribute values.
    keys that correspond to dictionaries go to new groups, recursively
    :param parent_group: parent group
    :param dic: dictionary
    :param name: name for the new group
    :return:
    """
    # group parent group or path, dict dictionary, name name of new group
    logger.debug('Translating dictionary key {} into its own group'.format(name))
    try:
        group = parent_group.create_group(name)
    except ValueError as err:
        logger.debug(err)
        if 'Name already exists' in err.args[0]:
            if replace:
                logger.debug('Group {} already exists; replacing'.format(name))
                group = parent_group.require_group(name)
            else:
                logger.debug('Group {} already exists; skipping'.format(name))
                return

    for key, item in dic.items():
        if not isinstance(item, dict):
            try:
                item = dict_2_attr_translator(item)
                group.attrs.create(key, item)
            except ValueError:
                logger.info("Wrong type error for key {}, setting None".format(type(item)))
                group.attrs.create(key, None)
        else:
            dict_2_group(group, item, key)


def attr_2_dict_translator(value):
    """
    :param value: value of an attribute
    :return: out_value for a dictionary
    """
    if type(value) is str and value == 'None':
        out_value = None
    else:
        out_value = value
    return out_value


def obj_attrs_2_dict_translator(h5obj):
    dic = {}
    for attr, value in h5obj.attrs.items():
        try:
            # logger.debug('attr {}'.format(attr))
            dic[attr] = attr_2_dict_translator(value)
        except ValueError:
            logger.warning("Could not translate value for attribute {}".format(attr))
            dic[attr] = None
    return dic


def group_2_dict(parent_dic, group, key_name):
    """
    Recursively dumps a group into a dictionary.
    attributes go to keys, values go to item values.
    subgroups to new dictionaries, recursively
    :param parent_dic: parent dictionary where to
    :param group: group to translate
    :param key_name: key of the dictionary to create for this group
    :return:
    """
    # enter a group with the attributes and lay it in a dictionary as key 'key_name'
    logger.debug('Translating group {} into its own dictionary'.format(key_name))
    parent_dic[key_name] = dict()
    dic = parent_dic[key_name]
    for attr, value in group.attrs.items():
        try:
            # logger.debug('attr {}'.format(attr))
            dic[attr] = attr_2_dict_translator(value)
        except ValueError:
            logger.warning("Could not translate value for attribute {}".format(attr))
            dic[attr] = None

    for subgroup_name, subgroup_obj in group.items():
        logger.debug('Subgroup {}'.format(subgroup_name))
        try:
            assert(isinstance(subgroup_obj, h5py.Group))
            # logger.info('Ok subgroup {}'.format(subgroup_name))
            group_2_dict(dic, subgroup_obj, subgroup_name)
        except AssertionError:
            try:
                assert (isinstance(subgroup_obj, h5py.Dataset))
                # logger.info('Ok dataset {}'.format(subgroup_name))
                dic[subgroup_name] = obj_attrs_2_dict_translator(subgroup_obj)
                # logger.info('Translated attrs of dataset {}'.format(subgroup_name))
            except:
                raise

    return parent_dic

def dset_2_arr(dset: h5py.Dataset) -> np.array:
    #raise NotImplemented
    if dset.size == 0:
        data = np.empty(0)
    else:
        data = dset
    return data


def insert_table(group, table, name, attr_dict=None):
    dset = group.create_dataset(name, data=table)
    if attr_dict is not None:
        append_atrributes(dset, attr_dict)
    return dset


def insert_group(parent_group, name, attr_dict_list=None):
    new_group = parent_group.create_group(name)
    if attr_dict_list is not None:
        append_atrributes(new_group, attr_dict_list)
    return new_group


def append_atrributes(h5obj, attr_dict):
    """
    Write a dictionary (no recursion) as a list of attributes belonging to an h5 object (dataset/group)
    :param h5obj: Group or Dataset h5py object.
    :param attr_dict: Dictionary (with no dictionaries)
    :return:
    """
    for key, item in attr_dict.items():
        #logger.info('key: {}'.format(key))
        #logger.info(item['name'] + ' {0} - {1}'.format(item['data'], item['dtype']))
        if isinstance(key, dict):
            logger.warning('Skipping sub-dictionary {0} in appending attributes of {1}'.format(key, h5obj.name))
            item = 'Error'
        h5obj.attrs.create(key, dict_2_attr_translator(item))

def bouts_from_h5(bout_h5_path, root_grp_name: str='bout_gpfa', exclude_dset=['spk_arr']):
    # open a h5 file.
    # read the group for the dataframe
    # append the bout metada dict keys to the group
    # for each group (bout)
        # create a dataframe with the salars from the attributes, and 
    # create a dataframe with the salars from the attributes, and
        # create a dataframe with the salars from the attributes, and 
        # pointers to the datasets for the tables
    logger.info('Loading bouts with their gpfa fits from {}'.format(bout_h5_path))
    all_bout_df = pd.DataFrame()
    with h5py.File(bout_h5_path,'r') as f:
        df_grp = f[root_grp_name]
        grp_attr_dict = obj_attrs_2_dict_translator(df_grp)
        # each key in the df_grp is a group, and each group is a bout
        #bout_idx_list = np.array(df_grp.keys())
        #logger.info(df_grp.keys())
        for bout_idx in df_grp.keys():
            #logger.info('bout_idx: {}'.format(bout_idx))
            bout_grp = df_grp[bout_idx]
            #logger.info('bout_grp {}'.format(bout_grp))
            bout_attr_dict = obj_attrs_2_dict_translator(bout_grp)
            bout_data_dict = {}
            for k in bout_grp.keys():
                if not (k in exclude_dset):
                    #logger.info('data key {}'.format(k))
                    bout_data_dict[k] =  bout_grp[k][:]
            
            bout_data_dict.update(bout_attr_dict)

            bout_df = pd.DataFrame(pd.Series(bout_data_dict)).T
            bout_df['bout_idx'] = bout_idx
            all_bout_df = pd.concat([all_bout_df, bout_df])
    
    all_bout_df.reset_index(inplace=True)
    return all_bout_df, grp_attr_dict

def bouts_to_h5(bout_df: pd.DataFrame, bout_dict: dict, dset_keys: list, attr_keys: list, bout_dict_keys: list, 
                h5_path: str,
               root_grp_name: str='bout_gpfa'):
    
    # make a h5 file.
    # create a group for the dataframe
    # append the bout metada dict keys to the group
    # for each row (bout)
        # create a group with the index of the bout
        # apend attr_keys with values for the bout as attributes (mostly scalars)
        # create datasets for all the long arrays that contain data over time

    with h5py.File(h5_path,'w') as f:
        df_grp = f.create_group(root_grp_name)
        grp_attr_dict = {k:bout_dict[k] for k in bout_dict_keys}
        append_atrributes(df_grp, grp_attr_dict)
        #dset_keys = ['mic_arr', 'pre_arr']
        logger.info('dict attributes done')
        
        for idx, row in enumerate(bout_df.itertuples()):
            logger.info('bout group {}'.format(idx))
            bout_grp = df_grp.create_group(str(idx))

            [bout_grp.create_dataset(k, data=getattr(row, k)) for k in dset_keys]

            bout_attr_dict = {k:getattr(row, k) for k in attr_keys}
            append_atrributes(bout_grp, bout_attr_dict)
            
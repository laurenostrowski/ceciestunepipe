import os
import sys
import glob
import logging
import shutil
import pickle


logger = logging.getLogger('ceciestunepipe.util.fileutil')


def makedirs(dir_path, exist_ok=True, mode=0o777):
    try:
        original_mask = os.umask(000)
        os.makedirs(dir_path, exist_ok=exist_ok, mode=mode)
    finally:
        os.umask(original_mask)

def chmod(file_path, mode=0o777):
    try:
        original_mask = os.umask(000)
        os.chmod(file_path, mode)
    finally:
        os.umask(original_mask)

def glob_except(path:str, glob_str:str='*.*', exclude_list=[]):
    all_files = glob.glob(os.path.join(path, glob_str))
    except_list = [glob.glob(os.path.join(path, s)) for s in exclude_list]
    flat_except = [f for sublist in except_list for f in sublist]
    some_files = [f for f in all_files if f not in flat_except]
    return some_files, flat_except

# copy path to dest, skip f existed
def safe_copy(src_path: str, dest_path: str):
    if not os.path.exists(dest_path):
        shutil.copyfile(src_path, dest_path)
    else:
        logger.info('file {} already existed, nuttin to do'.format(dest_path))

# append one binary file to another, chunked
def append_binary(src_path: str, dest_path: str, chunk_size: int=4194304):
    logger.info('Appending binaries {} -> {}'.format(src_path, dest_path))
    with open(dest_path, "ab") as dest_file, open(src_path, "rb") as src_file:
        shutil.copyfileobj(src_file, dest_file, chunk_size)

def save_pickle(obj, pickle_path):
    with open(pickle_path, 'wb') as fh:
        pickle.dump(obj, fh)
    chmod(pickle_path, mode=0o777)

def get_path_parts(file_path: str):
    folders = []
    while 1:
        file_path, folder = os.path.split(file_path)

        if folder != "":
            folders.append(folder)
        elif file_path != "":
            folders.append(file_path)

            break

    folders.reverse()
    return folders
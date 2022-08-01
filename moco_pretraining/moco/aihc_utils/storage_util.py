import os
import datetime

from pathlib import Path
import getpass

import getpass

if str(getpass.getuser()) == 'endiqq':
    # STORAGE_ROOT = Path('/home/jby/chexpert_experiments')
    STORAGE_ROOT = Path('self-learning/logdir')
else:
    STORAGE_ROOT = Path('/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments')


def get_storage_folder(exp_name, exp_type):

    try:
        jobid = os.environ["SLURM_JOB_ID"]
    except:
        jobid = None

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # username = str(getpass.getuser())

    fname = f'{exp_name}_{exp_type}_{datestr}_SLURM{jobid}' if jobid is not None else f'{exp_name}_{exp_type}_{datestr}'

    # path_name = STORAGE_ROOT / username / fname
    path_name = STORAGE_ROOT / fname
    os.makedirs(path_name)

    print(f'Experiment storage is at {fname}')
    return path_name


def get_storage_sub_folder(fname, ratio, iteration):
    # print (STORAGE_ROOT, fname)
    # path_name = STORAGE_ROOT / username / fname
    # sub_fname = str(fname).split('/')[-1]
    # print (sub_fname)
    path_name = fname / f'train_{ratio}_{iteration}'
    os.makedirs(path_name, exist_ok = True)

    print(f'Experiment storage is at {fname}')
    return path_name

def get_storage_sub_folder_acc(fname, ratio, iteration):
    # print (STORAGE_ROOT, fname)
    # path_name = STORAGE_ROOT / username / fname
    # sub_fname = str(fname).split('/')[-1]
    # print (sub_fname)
    path_name = fname / f'train_{ratio}_{iteration}_acc'
    os.makedirs(path_name, exist_ok = True)

    print(f'Experiment storage is at {fname}')
    return path_name
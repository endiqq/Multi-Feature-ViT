# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:44:32 2021

@author: RU
"""

from collections import defaultdict
import copy
import os
import pprint as pp
import random
import re
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

random.seed(0)



# def move_to_category(category, folder, df):
#     dst_path = Path(destination_root) / 'data' / folder / category
#     os.makedirs(str(dst_path), exist_ok=True)

#     df_path = Path(destination_root) / 'moving_logs' / f'{folder}_log.csv'
#     # TODO: JBY The splitting part is a bit of a hack, but welp, too lazy to fix
#     os.makedirs(str(Path(destination_root) / 'moving_logs' / folder.split('/')[0])
#                 , exist_ok=True)
#     df.to_csv(df_path)

#     new_paths = []
#     for i, row in tqdm(df.iterrows()):
#         fname = row['Path']
#         splitted_fname = fname.split('/')
#         new_fname = '_'.join(splitted_fname[-3:])

#         dst_fname = dst_path / new_fname

#         desired_path = Path(root_path / '/'.join(splitted_fname[1:]))
#         if not dst_fname.exists():
#             # os.symlink(str(desired_path), dst_fname)
#             shutil.copy(str(desired_path), dst_fname)
#         else:
#             # Do nothing for now
#             pass
            
#         new_paths.append(dst_fname)

#     return new_paths

def split_df(df, disease_name):
    '''Split a ground truth dataframe into no symptom and disease,
        Uncertain labels are treated as disease

        Note that we are NOT handling "No Finding" but only wrt a disease
    '''
    # healthy = df[df['No Finding'] == 1]
    no_sym = df[(df[disease_name] == 0) | (df[disease_name].isnull().values)]
    disease = df[(df[disease_name] == 1) | (df[disease_name] == -1)]
    print(f'Original: {str(len(df)).ljust(8)}\t\tNo Symptom: {len(no_sym)}, {disease_name}: {len(disease)}')

    assert len(df) == len(no_sym) + len(disease)
    return no_sym, disease



# parameters
ALL_SEMI_RATIO =  [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

SEMI_ITERATIONS = { 0.001: 5,
                    0.01: 5,
                    0.1: 5,
                    0.2: 5,
                    0.3: 5,
                    0.4: 5,
                    0.5: 5,
                    1: 1
                }

fine_tune_ratio_list = ALL_SEMI_RATIO
print(f'Fine tuning ratios are {fine_tune_ratio_list}')

root_folder = ['CheXpert-v1.0-small/CheXpert-v1.0-small']#,
               #'CheXpert_Enh/CheXpert-v1.0-small']
disease_name =  "Pleural Effusion"
disease_short_name = disease_name.replace(' ', '_').lower()
# img_type = ['CXR', 'Enh']
actual_val_ratio = 0.1


for idx, f in enumerate(root_folder):
    root_path = Path(f)
    
    ### Use the original valid as test ####
    print('===== Use valid as test =====')
    truth_csv = root_path / f'valid.csv'
    truth_df = pd.read_csv(truth_csv)
        
    no_sym, disease = split_df(truth_df, disease_name)
    
    total = pd.concat([no_sym, disease])
    # total.to_csv('test_'+img_type[idx]+'_'+f.split('/')[-1]+'.csv')
    total.to_csv('test_'+f.split('/')[-1]+'.csv')
    
    # disease.to_csv('test_'+img_type[idx]+'_'+f.split('/')[-1]+'.csv')
# move_to_category('no_sym', 'test', no_sym)
# move_to_category(disease_short_name, 'test', disease)

for idx, f in enumerate(root_folder):
    root_path = Path(f)

    print('===== Splitting for validation and fine tuning and training =====')
    truth_csv = root_path / f'train.csv'
    truth_df = pd.read_csv(truth_csv)
    
    # This way we ensure that lower semi-supervised ratios are subset of the higher ratio ones
    ratios = ALL_SEMI_RATIO + [1]
    
    actual_valid_rows = []
    fine_tune_train_rows = {s: [[] for it in range(SEMI_ITERATIONS[s])]
                               for s in ratios}
    fine_tune_tune_rows = {s: [[] for it in range(SEMI_ITERATIONS[s])]
                               for s in ratios}
    
    for stuff in tqdm(truth_df.iterrows()):
        i, row = stuff
        
        if random.random()< actual_val_ratio:
            actual_valid_rows.append(row)
        else:
            for s in ratios:
                for it in range(SEMI_ITERATIONS[s]):
                    rnd = random.random()
                    if rnd < s:
                        fine_tune_tune_rows[s][it].append(row)
                    else:
                        fine_tune_train_rows[s][it].append(row)

    # check number of each draw

    for i, s in enumerate(ratios):
        for it in range(SEMI_ITERATIONS[s]):
            print ('{}/{} = {}'.format(s,it,np.sum([len(actual_valid_rows),
                                                    len(fine_tune_train_rows[s][it]),
                                                    len(fine_tune_tune_rows[s][it])])))
            print (len(actual_valid_rows))
            print ('{}/{} = {}/{}'.format(s,it,len(fine_tune_train_rows[s][it]),
                                              len(fine_tune_tune_rows[s][it])))

    
    for i, s in enumerate(ratios):
        for it in range(SEMI_ITERATIONS[s]):
            train_df = pd.DataFrame(fine_tune_train_rows[s][it])
            finetune_df = pd.DataFrame(fine_tune_tune_rows[s][it])
            
            train_df.to_csv('train_'+str(s)+'_'+str(it)+'_'
                            +f.split('/')[-1]+'.csv')
            finetune_df.to_csv('finetune_'+str(s)+'_'+str(it)+'_'
                               +f.split('/')[-1]+'.csv')
            
    valid_df = pd.DataFrame(actual_valid_rows)
    valid_df.to_csv('valid_' +f.split('/')[-1]+'.csv')
            
    # valid_df = pd.DataFrame()

# check number of each draw
# import numpy as np
# for i, s in enumerate(ratios):
#     for it in range(SEMI_ITERATIONS[s]):
#         print ('{}/{} = {}'.format(s,it,np.sum([len(actual_valid_rows),
#                                                len(fine_tune_train_rows[s][it]),
#                                                len(fine_tune_tune_rows[s][it])])))


# for patient in structure:
#     if random.random() < actual_val_ratio:
#         assign_all_of_patient(structure, patient, actual_valid_ilocs)
#     else:
#         for i, fine_tune_ratio in enumerate(fine_tune_ratio_list):
#             if random.random() < fine_tune_ratio:
#                 fine_tune_tune_ilocs[i] = assign_all_of_patient(structure, patient, fine_tune_tune_ilocs[i])
#             # else:
#         fine_tune_train_ilocs = assign_all_of_patient(structure, patient, fine_tune_train_ilocs)

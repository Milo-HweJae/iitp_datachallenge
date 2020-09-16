# -*- coding:utf-8 -*-

import pandas as pd

from glob import glob

import numpy as np
import sklearn.metrics as metrics


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

def number_to_class(predict, threshold):
    label = []
    for i in range(len(predict)):
        if predict[i] < threshold:
            label.append('Normal')
        else:
            label.append('Attack')
    
    return label

def save_submission(DATA_PATH, pred, is_drive=True):
    
    submit_path = DATA_PATH + './제출용'
    
    if is_drive == True:
        submit_drive = sorted([x for x in glob(submit_path + "./*D_pred*.csv")])
        submit_df_drive = dataframe_from_csvs(submit_drive)
        submit_df_drive['Class'] = pred
        submit_df_drive.to_csv(submit_path + "./submission_d.csv")
    
    else:
        submit_stay = sorted([x for x in glob(submit_path + "./*S_pred*.csv")])
        submit_df_stay = dataframe_from_csvs(submit_stay)
        submit_df_stay['Class'] = pred
        submit_df_stay.to_csv(submit_path + "./submission_s.csv")
        
def scoring(DATA_PATH):
    
    score_path = DATA_PATH + './답안지'
    submit_path = DATA_PATH + './제출용'
    
    score_drive = sorted([x for x in glob(score_path + "./*D_pred*.csv")])
    score_stay = sorted([x for x in glob(score_path + "./*S_pred*.csv")])

    submit_drive = sorted([x for x in glob(submit_path + "./submission_d.csv")])
    submit_stay = sorted([x for x in glob(submit_path + "./submission_s.csv")])
    
    print ('[+] Loading score dataset')
    score_df_drive = dataframe_from_csvs(score_drive)
    score_df_stay = dataframe_from_csvs(score_stay)
    
    print ('[+] Loading submit dataset')
    submit_df_drive = dataframe_from_csvs(submit_drive)
    submit_df_stay = dataframe_from_csvs(submit_stay)
    
    answer_drive = np.asarray(score_df_drive['Class'])
    answer_stay = np.asarray(score_df_stay['Class'])
    
    submit_drive = np.asarray(submit_df_drive['Class'])
    submit_stay = np.asarray(submit_df_stay['Class'])
    
    for i in range(len(answer_drive)):
        if answer_drive[i] == 'Normal':
            answer_drive[i] = 0
        else:
            answer_drive[i] = 1
    
    for i in range(len(answer_stay)):
        if answer_stay[i] == 'Normal':
            answer_stay[i] = 0
        else:
            answer_stay[i] = 1
    
    for i in range(len(submit_drive)):
        if submit_drive[i] == 'Normal':
            submit_drive[i] = 0
        else:
            submit_drive[i] = 1
    
    for i in range(len(submit_stay)):
        if submit_stay[i] == 'Normal':
            submit_stay[i] = 0
        else:
            submit_stay[i] = 1            
    
    answer_drive = answer_drive.tolist()
    answer_stay = answer_stay.tolist()
    
    submit_drive = submit_drive.tolist()
    submit_stay = submit_stay.tolist()
    
    print('######drive score######')
    print('accuracy : ', metrics.accuracy_score(answer_drive, submit_drive))
    print('precision : ', metrics.precision_score(answer_drive, submit_drive))
    print('recall : ', metrics.recall_score(answer_drive, submit_drive))
    print('f1 : ', metrics.f1_score(answer_drive, submit_drive))
    print('\n')
    
    print('######stay score######')
    print('accuracy : ', metrics.accuracy_score(answer_stay, submit_stay))
    print('precision : ', metrics.precision_score(answer_stay, submit_stay))
    print('recall : ', metrics.recall_score(answer_stay, submit_stay))
    print('f1 : ', metrics.f1_score(answer_stay, submit_stay))  
    print('\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
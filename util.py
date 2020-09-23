# -*- coding:utf-8 -*-

import pandas as pd

from glob import glob

import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    # temp = pd.DataFrame(dataframe_from_csv)
    return pd.concat([dataframe_from_csv(x) for x in targets], ignore_index='True')

def number_to_class(predict, data, threshold):
    label = []
    for i in range(len(predict)):
        # loss = (predict[i] - data[i])**2
        loss = predict[i]
        if loss < threshold:
            label.append('Normal')
        else:
            label.append('Attack')
    
    return label

def save_submission(DATA_PATH, pred, past_history, is_drive=True):
    
    submit_path = DATA_PATH + './제출용'
    
    if is_drive == True:
        submit_drive = sorted([x for x in glob(submit_path + "./Cybersecurity_Car_Hacking_D_prediction.csv")])
        submit_df_drive = dataframe_from_csvs(submit_drive)
        for i in range(past_history):
            pred.insert(0,'Normal')
        submit_df_drive['Class'] = pred
        submit_df_drive.to_csv(submit_path + "./submission_d.csv")
    
    else:
        submit_stay = sorted([x for x in glob(submit_path + "./Cybersecurity_Car_Hacking_S_prediction.csv")])
        submit_df_stay = dataframe_from_csvs(submit_stay)
        for i in range(past_history):
            pred.insert(0,'Normal')
        submit_df_stay['Class'] = pred
        submit_df_stay.to_csv(submit_path + "./submission_s.csv")
        
def scoring(DATA_PATH):
    
    score_path = DATA_PATH + './답안지'
    submit_path = DATA_PATH + './제출용'
    
    score_drive = sorted([x for x in glob(score_path + "./Answer_Cybersecurity_Car_Hacking_D_prediction_.csv")])
    score_stay = sorted([x for x in glob(score_path + "./Answer_Cybersecurity_Car_Hacking_S_prediction_.csv")])

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
    
    # submit_drive = np.insert(submit_drive,0,np.zeros(100))
    # submit_stay = np.insert(submit_stay,0,np.zeros(100))
    
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
        if submit_drive[i] == 'Normal' or submit_drive[i] == 0:
            submit_drive[i] = 0
        else:
            submit_drive[i] = 1
    
    for i in range(len(submit_stay)):
        if submit_stay[i] == 'Normal' or submit_stay[i] == 0:
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
    
    y_pred = submit_drive
    y_true = answer_drive
    tn, fp, fn,tp = confusion_matrix(y_true,y_pred).ravel()
    print(tp,tn,fn,fp)
    y_pred = submit_stay
    y_true = answer_stay
    tn, fp, fn,tp = confusion_matrix(y_true,y_pred).ravel()
    print(tp,tn,fn,fp)
    
    
    
    
    
    
    
    
    
    
    
    
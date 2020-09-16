# -*- coding:utf-8 -*-

from util import dataframe_from_csvs
from glob import glob
import numpy as np

def CAR_preprocessing(DATA_PATH):
    print('[+] Start preprocession')
    
    train_path = DATA_PATH + './학습용'
    submit_path = DATA_PATH + './제출용'

    train_drive = sorted([x for x in glob(train_path + "./*D_training*.csv")])
    train_stay = sorted([x for x in glob(train_path + "./*S_training*.csv")])

    submit_drive = sorted([x for x in glob(submit_path + "./*D_pred*.csv")])
    submit_stay = sorted([x for x in glob(submit_path + "./*S_pred*.csv")])

  
    print ('[+] Loading train dataset')
    train_df_drive = dataframe_from_csvs(train_drive)
    train_df_stay = dataframe_from_csvs(train_stay)
    
    print ('[+] Loading submit dataset')
    submit_df_drive = dataframe_from_csvs(submit_drive)
    submit_df_stay = dataframe_from_csvs(submit_stay)
    
    
    # train_d_data, train_d_label
    train_df_drive_ = train_df_drive.iloc[:,1:5]
    
    train_df_drive, train_d_label = split_target(train_df_drive_)
    train_df_drive = np.asarray(train_df_drive)
    
    train_d_data = []
    for i in range(len(train_df_drive)):
        tmp = train_df_drive[i][2].split(" ")
        
        train_d_data.append(int(train_df_drive[i][0].encode("utf-8").hex())) 
        train_d_data.append(train_df_drive[i][1])
        number=''
        for j in range(len(tmp)):
            number = number + tmp[j].encode("utf-8").hex()
        train_d_data.append(int(number))    
    
    train_d_label = np.asarray(train_d_label)
    
    train_d_data = np.asarray(train_d_data)
    train_d_data = train_d_data.reshape(-1,3)
    train_d_data = train_d_data[::-1]
    train_d_data = train_d_data.astype(np.float32)        
    
    # train_s_data, train_s_label
    train_df_stay_ = train_df_stay.iloc[:,1:5]
    
    train_df_stay, train_s_label = split_target(train_df_stay_)
    train_df_stay = np.asarray(train_df_stay)

    train_s_data = []
    for i in range(len(train_df_stay)):
        tmp = train_df_drive[i][2].split(" ")
        
        train_s_data.append(int(train_df_stay[i][0].encode("utf-8").hex())) 
        train_s_data.append(train_df_stay[i][1])
        number=''
        for j in range(len(tmp)):
            number = number + tmp[j].encode("utf-8").hex()
        train_s_data.append(int(number))    
    
    train_s_label = np.asarray(train_s_label)
    
    train_s_data = np.asarray(train_s_data)
    train_s_data = train_s_data.reshape(-1,3)
    train_s_data = train_s_data[::-1]
    train_s_data = train_s_data.astype(np.float32) 
    
    # submit_d_data
    submit_df_drive = submit_df_drive.iloc[:,2:5]
    
    submit_df_drive = np.asarray(submit_df_drive)
    
    submit_d_data = []
    for i in range(len(submit_df_drive)):
        tmp = submit_df_drive[i][2].split(" ")
        
        submit_d_data.append(int(submit_df_drive[i][0].encode("utf-8").hex())) 
        submit_d_data.append(submit_df_drive[i][1])
        number=''
        for j in range(len(tmp)):
            number = number + tmp[j].encode("utf-8").hex()
        submit_d_data.append(int(number))    
    
    submit_d_data = np.asarray(submit_d_data)
    submit_d_data = submit_d_data.reshape(-1,3)
    submit_d_data = submit_d_data[::-1]
    submit_d_data = submit_d_data.astype(np.float32) 
    
    # submit_s_data
    submit_df_stay = submit_df_stay.iloc[:,2:5]
    
    submit_df_stay = np.asarray(submit_df_stay)
    
    submit_s_data = []
    for i in range(len(submit_df_stay)):
        tmp = submit_df_stay[i][2].split(" ")
        
        submit_s_data.append(int(submit_df_stay[i][0].encode("utf-8").hex())) 
        submit_s_data.append(submit_df_stay[i][1])
        number=''
        for j in range(len(tmp)):
            number = number + tmp[j].encode("utf-8").hex()
        submit_s_data.append(int(number))    
    
    submit_s_data = np.asarray(submit_s_data)
    submit_s_data = submit_s_data.reshape(-1,3)
    submit_s_data = submit_s_data[::-1]
    submit_s_data = submit_s_data.astype(np.float32) 

    
    return train_d_data, train_d_label, train_s_data, train_s_label, submit_d_data, submit_s_data

def split_target(data):
    label_tmp = data.iloc[1:, 3:]
    data = data.iloc[1:, :3]
    
    label = []
    for item in label_tmp.iterrows():
        if item[1][0] == 'Normal':
            label.append(0)
        else:
            label.append(1)
            
    label = np.asarray(label)
    
    return data, label
# -*- coding:utf-8 -*-

from util import dataframe_from_csvs
from glob import glob
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd

def load_data(DATA_PATH):
    print('[+] Start preprocession')
    
    train_path = DATA_PATH + './학습용'
    submit_path = DATA_PATH + './제출용'

    train_drive = sorted([x for x in glob(train_path + "./*D_training*.csv")])
    train_stay = sorted([x for x in glob(train_path + "./*S_training*.csv")])

    submit_drive = sorted([x for x in glob(submit_path + "./Cybersecurity_Car_Hacking_D_prediction.csv")])
    submit_stay = sorted([x for x in glob(submit_path + "./Cybersecurity_Car_Hacking_S_prediction.csv")])

  
    print ('[+] Loading train dataset')
    train_df_drive = dataframe_from_csvs(train_drive)
    train_df_stay = dataframe_from_csvs(train_stay)
    
    print ('[+] Loading submit dataset')
    submit_df_drive = dataframe_from_csvs(submit_drive)
    submit_df_stay = dataframe_from_csvs(submit_stay)
    
    # print(train_df_drive.index)
    # train_d_data, train_d_label
    # train_d_data.index = train_df_drive['index_col']
    # train_d_data = train_d_data.replace('Normal', '0')
    # train_d_data = train_d_data.replace('Attack', '1')
    # train_d_data = train_d_data.apply(pd.to_numeric)
    
    train_d_data = train_df_drive.Arbitration_ID.apply(lambda x: int(x,16))
    # train_d_data = train_d_data_temp
    
    # train_s_data.index = train_df_stay['index_col']
    # train_s_data = train_s_data.replace('Normal', '0')
    # train_s_data = train_s_data.replace('Attack', '1')
    # train_s_data = train_s_data.apply(pd.to_numeric)
    
    train_s_data = train_df_stay.Arbitration_ID.apply(lambda x: int(x,16))        
    # train_s_data = train_s_data_temp
    
    submit_d_data = submit_df_drive.Arbitration_ID.apply(lambda x: int(x,16))
    
    submit_s_data = submit_df_stay.Arbitration_ID.apply(lambda x: int(x,16))
    
    return train_d_data[:1700000], train_s_data[:1700000], submit_d_data, submit_s_data

def preprocessing_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
  
    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size
  
    for i in range(start_index, end_index):
      indices = list(range(i-history_size, i))
      # Reshape data from (history_size,) to (history_size, 1)
      data.append(np.reshape(dataset[indices], (history_size,)))
      labels.append(dataset[i+target_size])
    data = np.array(data)
    labels = np.array(labels)
    data = data.reshape(-1,1,history_size)
    labels = labels.reshape(-1,1)
    return data, labels

def preprocessing_data_submit(dataset, start_index, end_index, history_size, target_size):
    data = []
    # labels = []
  
    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size
  
    for i in range(start_index, end_index):
      indices = list(range(i-history_size, i))
      # Reshape data from (history_size,) to (history_size, 1)
      data.append(np.reshape(dataset[indices], (history_size,)))
      # labels.append(dataset[i+target_size])
    data = np.array(data)
    # labels = np.array(labels)
    data = data.reshape(-1,1,history_size)
    # labels = labels.reshape(-1,1)
    return data
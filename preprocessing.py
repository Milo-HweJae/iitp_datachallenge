# -*- coding:utf-8 -*-

from util import dataframe_from_csvs
from glob import glob
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
from keras.utils import np_utils

def replace(data):
    can_id = ['340','2B0','4A4','164','260','251','140','130','153','220','7D8','7D0','7DC','7D4','7CC',
              '7C4','57F','43','07F','5BE','4A2','4A7','53F','4A9','5CD','5B0','52A','5A6','53B','4CB','50E',
              '412','410','48C','559','572','49F','44E','553','544','4C9','541','50A','483','500','495','479',
              '50C','507','568','593','436','58B','48A','53E','42D','492','485','520','381','490','484','470',
              '453','394','387','329','563','47F','260','220','153','2B0','251','4F1','391','389','421','420',
              '38D','386','140','130','164','368','367','366','356']
    for i in range(len(can_id)):
        data = data.str.replace(pat=can_id[i],repl=str(i+1),regex=False)
    data = pd.to_numeric(data)
    
    return data

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
    
    train_d_data = train_df_drive.Arbitration_ID
    train_d_data = replace(train_d_data)
    train_s_data = train_df_stay.Arbitration_ID       
    train_s_data = replace(train_s_data)
    submit_d_data = submit_df_drive.Arbitration_ID
    submit_d_data = replace(submit_d_data)
    submit_s_data = submit_df_stay.Arbitration_ID
    submit_s_data = replace(submit_s_data)
    
    return train_d_data, train_s_data, submit_d_data, submit_s_data

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
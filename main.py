# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


from util import dataframe_from_csv, dataframe_from_csvs
from preprocessing import CAR_preprocessing

if __name__ == '__main__':

    DATA_PATH = './데이터셋'
    EPOCH = 0
    BATCH_SIZE = 256
    THRESHOLD = 0.01
    
    CAR_preprocessing(DATA_PATH)
    # print (train_data, submit_data)

    # train_data, test_data = dataframe_from_csvs(DATA_PATH)
    
    # model = SomeThing(train_data, shape, BATCH_SIZE, EPOCH, load_only=False, plot=False)

    

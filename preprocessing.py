# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from util import dataframe_from_csv, dataframe_from_csvs
import re

def CAR_preprocessing(DATA_PATH):
    print('[+] Start preprocession')
    
    train_path = DATA_PATH + './학습용'
    submit_path = DATA_PATH + './제출용'

    train_drive = sorted([x for x in Path(train_path).glob("*D_training*.csv")])
    train_stay = sorted([x for x in Path(train_path).glob("*S_training*.csv")])

    submit_drive = sorted([x for x in Path(submit_path).glob("*D_prediction*.csv")])
    submit_stay = sorted([x for x in Path(submit_path).glob("*S_prediction*.csv")])

  
    print ('[+] Loading train dataset')
    train_df_drive = dataframe_from_csvs(train_drive)
    train_df_drive = dataframe_from_csvs(train_stay)

    submit_df_drive = dataframe_from_csv(submit_drive)
    submit_df_stay = dataframe_from_csv(submit_stay)

    return

    
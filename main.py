# -*- coding:utf-8 -*-

from util import number_to_class, save_submission, scoring
from preprocessing import preprocessing_data, load_data, preprocessing_data_submit

from keras import optimizers
from keras.optimizers import SGD

from sklearn.preprocessing import RobustScaler

import tensorflow as tf


if __name__ == '__main__':
    
    DATA_PATH = './데이터셋'
    EPOCH = 1
    BATCH_SIZE = 2048
    BUFFER_SIZE = 10000
    DRIVE_THRESHOLD = 0.3
    STAY_THRESHOLD = 0.3
    past_history = 5
    future_target = 0
    TRAIN_SPLIT = 1500000
    EVALUATION_INTERVAL = 2000
    # preprocessing dataset
    train_d_data_, train_s_data_, submit_d_data_, submit_s_data_ = load_data(DATA_PATH)
    
    # # data save
    import pickle
    # save
    with open('train_d_data_.pkl','wb') as f:
        pickle.dump(train_d_data_, f)
        
    with open('train_s_data_.pkl','wb') as f:
        pickle.dump(train_s_data_, f)
        
    with open('submit_d_data_.pkl','wb') as f:
        pickle.dump(submit_d_data_, f)
    
    with open('submit_s_data_.pkl','wb') as f:
        pickle.dump(submit_s_data_, f)
    
    # pickle load    
    with open('train_d_data_.pkl','rb') as f:
        train_d_data_ = pickle.load(f)
        
    with open('train_s_data_.pkl','rb') as f:
        train_s_data_ = pickle.load(f)
        
    with open('submit_d_data_.pkl','rb') as f:
        submit_d_data_ = pickle.load(f)
    
    with open('submit_s_data_.pkl','rb') as f:
        submit_s_data_ = pickle.load(f)

    
    train_d_data, train_d_label = preprocessing_data(train_d_data_, 0, TRAIN_SPLIT, past_history, future_target)
    train_d_data_val, train_d_label_val = preprocessing_data(train_d_data_, TRAIN_SPLIT, None, past_history, future_target)
    train_s_data, train_s_label = preprocessing_data(train_s_data_, 0, TRAIN_SPLIT, past_history, future_target)
    train_s_data_val, train_s_label_val = preprocessing_data(train_s_data_, TRAIN_SPLIT, None, past_history, future_target)
    
    submit_d_data = preprocessing_data_submit(submit_d_data_, 0, None, past_history, future_target)
    submit_s_data = preprocessing_data_submit(submit_s_data_, 0, None, past_history, future_target)
    
    with open('train_d_data.pkl','wb') as f:
        pickle.dump(train_d_data, f)
    with open('train_d_label.pkl','wb') as f:
        pickle.dump(train_d_label, f)
    with open('train_d_data_val.pkl','wb') as f:
        pickle.dump(train_d_data_val, f)
    with open('train_d_label_val.pkl','wb') as f:
        pickle.dump(train_d_label_val, f)
    with open('train_s_data.pkl','wb') as f:
        pickle.dump(train_s_data, f)        
    with open('train_s_label.pkl','wb') as f:
        pickle.dump(train_s_label, f)
    with open('train_s_data_val.pkl','wb') as f:
        pickle.dump(train_s_data_val, f)
    with open('train_s_label_val.pkl','wb') as f:
        pickle.dump(train_s_label_val, f)
    with open('submit_d_data.pkl','wb') as f:
        pickle.dump(submit_d_data, f)
    with open('submit_s_data.pkl','wb') as f:
        pickle.dump(submit_s_data, f)
    
    with open('train_d_data.pkl','rb') as f:
        train_d_data = pickle.load(f)
    with open('train_d_label.pkl','rb') as f:
        train_d_label = pickle.load(f)
    with open('train_d_data_val.pkl','rb') as f:
        train_d_data_val = pickle.load(f)
    with open('train_d_label_val.pkl','rb') as f:
        train_d_label_val = pickle.load(f)
    with open('train_s_data.pkl','rb') as f:
        train_s_data = pickle.load(f)        
    with open('train_s_label.pkl','rb') as f:
        train_s_label = pickle.load(f)
    with open('train_s_data_val.pkl','rb') as f:
        train_s_data_val = pickle.load(f)
    with open('train_s_label_val.pkl','rb') as f:
        train_s_label_val = pickle.load(f)
    with open('submit_d_data.pkl','rb') as f:
        submit_d_data = pickle.load(f)
    with open('submit_s_data.pkl','rb') as f:
        submit_s_data = pickle.load(f)
    
    train_d = tf.data.Dataset.from_tensor_slices((train_d_data, train_d_label))
    train_d = train_d.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_d_val = tf.data.Dataset.from_tensor_slices((train_d_data_val, train_d_label_val))
    train_d_val = train_d_val.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    train_s = tf.data.Dataset.from_tensor_slices((train_s_data, train_s_label))
    train_s = train_s.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_s_val = tf.data.Dataset.from_tensor_slices((train_s_data_val, train_s_label_val))
    train_s_val = train_s_val.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    # make a model
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov = True)
    print('[+] Make model')
    drive_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(1,past_history)),
        # tf.keras.layers.Dense(100, activation = 'relu'),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dense(1, activation='softmax')
        ])
    drive_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    stay_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(1,past_history)),
        # tf.keras.layers.Dense(100, activation = 'relu'),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dense(1, activation='softmax')
        ])
    stay_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # model train
    print('[+] Train model')
    drive_model.fit(train_d, epochs = EPOCH, steps_per_epoch=EVALUATION_INTERVAL, validation_data = train_d_val, validation_steps=500)
    stay_model.fit(train_s, epochs = EPOCH, steps_per_epoch=EVALUATION_INTERVAL, validation_data = train_s_val, validation_steps=500)
    
    # predict
    drive_predict = drive_model.predict(submit_d_data)
    stay_predict = stay_model.predict(submit_s_data)
    
    # # change number to class_name
    # d_predict = number_to_class(drive_predict[:,0], submit_d_data_, DRIVE_THRESHOLD)
    # s_predict = number_to_class(stay_predict[:,0], submit_s_data_, STAY_THRESHOLD)
    
    # # record
    # print('[+] Record label')
    # save_submission(DATA_PATH, d_predict, past_history, is_drive = True)
    # save_submission(DATA_PATH, s_predict, past_history, is_drive = False)
    
    # # score
    # print('[+] score')
    # scoring(DATA_PATH)
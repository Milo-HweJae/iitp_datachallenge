# -*- coding:utf-8 -*-

from util import number_to_class, save_submission, scoring
from preprocessing import CAR_preprocessing

from model import make_model

from keras import optimizers
from keras.optimizers import SGD

from sklearn.preprocessing import RobustScaler

if __name__ == '__main__':
    
    DATA_PATH = './데이터셋'
    EPOCH = 1
    BATCH_SIZE = 1024
    DRIVE_THRESHOLD = 0.110907
    STATIC_THRESHOLD = 0.101258
    # preprocessing dataset
    # train_d_data, train_d_label, train_s_data, train_s_label, submit_d_data, submit_s_data = CAR_preprocessing(DATA_PATH)
    
    # train_d_data : 주행 중 학습 데이터셋의 id, bit, data
    # train_s_data : 정차 중 학습 데이터셋의 id, bit, data
    # train_d_label : 주행 중 학습 데이터셋의 공격 라벨, 정상 : 0, 공격 : 1
    # train_s_label : 정차 중 학습 데이터셋의 공격 라벨, 정상 : 0, 공격 : 1
    # submit_d_data : 주행 중 제출 데이터셋의 id, bit, data
    # submit_s_data : 정차 중 제출 데이터셋의 id, bit, data
    
    # save pickle
    
    import pickle

    # with open('train_d_data.pkl','wb') as f:
    #  	pickle.dump(train_d_data, f)
    
    # with open('train_d_label.pkl','wb') as f:
    #  	pickle.dump(train_d_label, f)
    
    # with open('train_s_data.pkl','wb') as f:
    #  	pickle.dump(train_s_data, f)
    
    # with open('train_s_label.pkl','wb') as f:
    #  	pickle.dump(train_s_label, f)
    
    # with open('submit_d_data.pkl','wb') as f:
    #  	pickle.dump(submit_d_data, f)
    
    # with open('submit_s_data.pkl','wb') as f:
    #  	pickle.dump(submit_s_data, f)
    
    with open('train_d_data.pkl','rb') as f:
    	train_d_data = pickle.load(f)
    
    with open('train_d_label.pkl','rb') as f:
    	train_d_label = pickle.load(f)
    
    with open('train_s_data.pkl','rb') as f:
    	train_s_data = pickle.load(f)
    
    with open('train_s_label.pkl','rb') as f:
    	train_s_label = pickle.load(f)
    
    with open('submit_d_data.pkl','rb') as f:
    	submit_d_data = pickle.load(f)
    
    with open('submit_s_data.pkl','rb') as f:
    	submit_s_data = pickle.load(f)
    
    # make a model
    print('[+] Make model')
    drive_model = make_model()
    static_model = make_model()
    
    # model compile
    drive_model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
    static_model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
    
    # normalization
    train_d_data = RobustScaler().fit_transform(train_d_data)
    train_s_data = RobustScaler().fit_transform(train_s_data)
    submit_d_data = RobustScaler().fit_transform(submit_d_data)
    submit_s_data = RobustScaler().fit_transform(submit_s_data)
    
    # model train
    print('[+] Train model')
    drive_model.fit(train_d_data, train_d_label, BATCH_SIZE, EPOCH, verbose=1)
    static_model.fit(train_s_data, train_s_label, BATCH_SIZE, EPOCH, verbose=1)
    
    # predict
    drive_predict = drive_model.predict(submit_d_data)
    static_predict = static_model.predict(submit_s_data)
    
    # change number to class_name
    d_predict = number_to_class(drive_predict, DRIVE_THRESHOLD)
    s_predict = number_to_class(static_predict, STATIC_THRESHOLD)
    
    # record
    print('[+] Record label')
    save_submission(DATA_PATH, d_predict, is_drive = True)
    save_submission(DATA_PATH, s_predict, is_drive = False)
    
    # score
    print('[+] score')
    scoring(DATA_PATH)
    
    
    
    
    
    
    
    
    
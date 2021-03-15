# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:46:52 2021

@author: Henry
"""

import pandas as pd
import keras
import tensorflow
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils

frequency = 256
window_time = 7
window_size = window_time * frequency


FallAllD = pd.read_hdf('FallAllD.h5', 'df')

data_wrist = []
label_wrist = []
subject_wrist = []
data_waist = []
label_waist = []
subject_waist = []
data_neck = []
label_neck = []
subject_neck = []

for i in range(len(FallAllD)):
    if(FallAllD['Device'][i] == 'Neck'):
        data_neck.append(FallAllD['Acc'][i]*0.000244)
        subject_neck.append(FallAllD['SubjectID'][i])
        if(FallAllD['ActivityID'][i]>=100):
            label_neck.append(1)
        else:
            label_neck.append(0)
    else:
        if (FallAllD['Device'][i] == 'Wrist'):
            data_wrist.append(FallAllD['Acc'][i]*0.000244)
            subject_wrist.append(FallAllD['SubjectID'][i])
            if(FallAllD['ActivityID'][i]>=100):
                label_wrist.append(np.uint8(1))
            else:
                label_wrist.append(np.uint8(0))
        else:
            data_waist.append(FallAllD['Acc'][i]*0.000244)
            subject_waist.append(FallAllD['SubjectID'][i])
            if(FallAllD['ActivityID'][i]>=100):
                    label_waist.append(1)
            else:
                    label_waist.append(0)
                    
label_neck = np.array(label_neck)
subject_neck = np.array(subject_neck)
data_neck = np.array(data_neck)
label_waist = np.array(label_waist)
subject_waist = np.array(subject_waist)
data_waist = np.array(data_waist)
label_wrist = np.array(label_wrist)
subject_wrist = np.array(subject_wrist)
data_wrist = np.array(data_wrist)


sliding_waist_data = []
sliding_waist_label = []
sliding_waist_subject = []

for data_num in range(len(data_waist)):

    #max_norm = max( np.sqrt(pow(data_waist[data_num][:,0],2) + pow(data_waist[data_num][:,1],2) + pow(data_waist[data_num][:,2],2) ) )
    if label_waist[data_num] == 1:
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size/2):np.uint(1.5*window_size-1),:])
        sliding_waist_label.append(np.uint8(1))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size):np.uint(2*window_size-1),:])
        sliding_waist_label.append(np.uint8(1))
        sliding_waist_subject.append(subject_waist[data_num])
    else:
        sliding_waist_data.append(data_waist[data_num][0:np.uint(window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size/2):np.uint(1.5*window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size):np.uint(2*window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_data.append(data_waist[data_num][np.uint(1.5*window_size):np.uint(2.5*window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        
        
sliding_waist_data = np.array(sliding_waist_data)
sliding_waist_label = np.array(sliding_waist_label)
sliding_waist_subject = np.array(sliding_waist_subject)
        
for fold in np.hstack([range(1,6),range(9,15)]):
    if(fold == 7 or fold ==8):
        continue
    
    test = (sliding_waist_subject == fold)
    train = ~test

    X_train = sliding_waist_data[train,:,:]
    Y_train = sliding_waist_label[train]
    Y_train_hot = np_utils.to_categorical(Y_train, 2)
    
    X_test = sliding_waist_data[test,:,:]
    Y_test = sliding_waist_label[test]
    Y_test_hot = np_utils.to_categorical(Y_test, 2)
    X_train = X_train.reshape(len(X_train),(window_size-1)*3) 
    X_test = X_test.reshape(len(X_test),(window_size-1)*3)
    
    model_fall = Sequential()
    model_fall.add(Dense(units = 64,input_dim=(window_size-1)*3))
    model_fall.add(Activation("relu"))
    model_fall.add(Dense(units = 64))
    model_fall.add(Activation("relu"))
    model_fall.add(Dense(units = 64))
    model_fall.add(Activation("relu"))
    model_fall.add(Dense(units = 2))
    model_fall.add(Activation("softmax"))

    model_fall.compile(
        loss = 'binary_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])
    model_fall.fit(
        X_train,
        Y_train_hot,
        batch_size = 50,
        epochs = 20,
        verbose = 1)

    score = model_fall.evaluate(X_test,Y_test_hot,verbose = 1)
    model_fall.save('keras.fall.model')
    print('loss:',score[0])
    print('accuracy',score[1])
    
    

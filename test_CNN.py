# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:43:21 2021

@author: Henry
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

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


def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.savefig('Confusion.png', dpi=150)
    plt.show()


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
        
simple_accuracy = []
simple_recall = []
simple_precision = []
simple_f1score = []


sliding_waist_label_hot =np.array([])
sliding_waist_label_hot = np_utils.to_categorical(sliding_waist_label)

for fold in [1,2,3,4,5,7,8,9,10,11,12,13,14,15]:
    print('Fold:',fold)
#    if(fold == 7 or fold ==8):
 #       continue
    
    test = (sliding_waist_subject == fold)
    train = ~test

    X_train = sliding_waist_data[train,:,:]
    Y_train = sliding_waist_label_hot[train]
    
    X_test = sliding_waist_data[test,:,:]
    Y_test = sliding_waist_label_hot[test]
    Y_test_gd = sliding_waist_label[test]
    X_train = X_train.reshape(len(X_train),(window_size-1),3,1) 
    X_test = X_test.reshape(len(X_test),(window_size-1),3,1)
    
    model_fall = Sequential()
    model_fall.add(Conv2D(32,(3,3),input_shape = ((window_size-1),3,1)))
    model_fall.add(MaxPooling2D(pool_size = (2,1)))
    model_fall.add(Conv2D(32,(3,1)))
    model_fall.add(MaxPooling2D(pool_size = (2,1)))
    model_fall.add(Flatten())
    model_fall.add(Dense(units = 64))
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
        Y_train,
        batch_size = 50,
        epochs = 20,
        verbose = 1)

    #model_fall.save('keras.fall.model')
    score = model_fall.evaluate(X_test,Y_test,verbose = 1)
    
    print('loss:',score[0])
    print('accuracy',score[1])
    
    y_pred_test = model_fall.predict(X_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    
    # max_y_test = np.argmax(y_test, axis=1)

    show_confusion_matrix(Y_test_gd, max_y_pred_test)
   

    simple_accuracy.append(accuracy_score(Y_test_gd, max_y_pred_test))
    simple_recall.append(recall_score(Y_test_gd, max_y_pred_test))
    simple_precision.append(precision_score(Y_test_gd, max_y_pred_test))
    simple_f1score.append(f1_score(Y_test_gd, max_y_pred_test))

    
    print(accuracy_score(Y_test_gd, max_y_pred_test))
    print(precision_score(Y_test_gd, max_y_pred_test))
    print(recall_score(Y_test_gd, max_y_pred_test))
    print(f1_score(Y_test_gd, max_y_pred_test))
    
    

overall_simple_accuracy = sum(simple_accuracy)/len(simple_accuracy)
overall_simple_recall = sum(simple_recall)/len(simple_recall)
overall_simple_precision = sum(simple_precision)/len(simple_precision)
overall_simple_f1score = sum(simple_f1score)/len(simple_f1score)



plot_model(model_fall, to_file='single_outputs.png')



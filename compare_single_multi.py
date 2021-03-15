# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:09:02 2021

@author: Henry
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Flatten
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from scipy.stats import ttest_rel


frequency = 256
window_time = 7
window_size = window_time * frequency

alpha_per = 0.5
num_dense_unit = 64

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

complex_fall_waist = []

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
                    complex_fall_waist.append(FallAllD['ActivityID'][i]-56)
            else:
                    label_waist.append(0)
                    complex_fall_waist.append(FallAllD['ActivityID'][i])
                    
label_neck = np.array(label_neck)
subject_neck = np.array(subject_neck)
data_neck = np.array(data_neck)
label_waist = np.array(label_waist)
subject_waist = np.array(subject_waist)
data_waist = np.array(data_waist)
label_wrist = np.array(label_wrist)
subject_wrist = np.array(subject_wrist)
data_wrist = np.array(data_wrist)
complex_fall_waist = np.array(complex_fall_waist)

sliding_waist_data = []
sliding_waist_label = []
sliding_waist_subject = []
sliding_waist_complex = []

for data_num in range(len(data_waist)):

    #max_norm = max( np.sqrt(pow(data_waist[data_num][:,0],2) + pow(data_waist[data_num][:,1],2) + pow(data_waist[data_num][:,2],2) ) )
    if label_waist[data_num] == 1:
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size/2):np.uint(1.5*window_size-1),:])
        sliding_waist_label.append(np.uint8(1))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_complex.append(complex_fall_waist[data_num])
        
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size):np.uint(2*window_size-1),:])
        sliding_waist_label.append(np.uint8(1))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_complex.append(complex_fall_waist[data_num])
    else:
        sliding_waist_data.append(data_waist[data_num][0:np.uint(window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_complex.append(complex_fall_waist[data_num])
        
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size/2):np.uint(1.5*window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_complex.append(complex_fall_waist[data_num])
        
        sliding_waist_data.append(data_waist[data_num][np.uint(window_size):np.uint(2*window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_complex.append(complex_fall_waist[data_num])
        
        sliding_waist_data.append(data_waist[data_num][np.uint(1.5*window_size):np.uint(2.5*window_size-1),:])
        sliding_waist_label.append(np.uint8(0))
        sliding_waist_subject.append(subject_waist[data_num])
        sliding_waist_complex.append(complex_fall_waist[data_num])
        
sliding_waist_data = np.array(sliding_waist_data)
sliding_waist_label = np.array(sliding_waist_label)
sliding_waist_subject = np.array(sliding_waist_subject)
sliding_waist_complex = np.array(sliding_waist_complex)

class_list = list(range(1,86))

single_accuracy = []
single_precision = []
single_recall = []
single_f1score = []

simple_accuracy = []
simple_recall = []
simple_precision = []
simple_f1score = []
complex_accuracy = []
complex_cm = []

sliding_waist_label_hot =np.array([])
sliding_waist_complex_hot = np.array([])

sliding_waist_label_hot = np_utils.to_categorical(sliding_waist_label)
sliding_waist_complex_hot = np_utils.to_categorical(sliding_waist_complex)


for fold in [1,2,3,4,5,7,8,9,10,11,12,13,14,15]:
    print('Fold:',fold)
    
    'train & test'
    test = (sliding_waist_subject == fold)
    train = ~test

    X_train = sliding_waist_data[train,:,:]
    Y_train = sliding_waist_label_hot[train]
    Z_train = sliding_waist_complex_hot[train]
    
    X_test = sliding_waist_data[test,:,:]
    Y_test = sliding_waist_label_hot[test]
    Z_test = sliding_waist_complex_hot[test]
    Y_test_gd = sliding_waist_label[test]
    Z_test_gd = sliding_waist_complex[test]
    
    X_train = X_train.reshape(len(X_train),(window_size-1),3,1) 
    X_test = X_test.reshape(len(X_test),(window_size-1),3,1)
    
    
    
    'build model'
    
    'single task'
    print('single task')
    model_single = Sequential()
    model_single.add(Conv2D(32,(3,3),input_shape = ((window_size-1),3,1)))
    model_single.add(MaxPooling2D(pool_size = (2,1)))
    model_single.add(Conv2D(32,(3,1)))
    model_single.add(MaxPooling2D(pool_size = (2,1)))
    model_single.add(Flatten())
    model_single.add(Dense(units = num_dense_unit))
    model_single.add(Activation("relu"))
    model_single.add(Dense(units = num_dense_unit))
    model_single.add(Activation("relu"))
    model_single.add(Dense(units = num_dense_unit))
    model_single.add(Activation("relu"))
    model_single.add(Dense(units = 2))
    model_single.add(Activation("softmax"))

    model_single.compile(
        loss = 'binary_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])
    model_single.fit(
        X_train,
        Y_train,
        batch_size = 50,
        epochs = 20,
        verbose = 1)

    #model_fall.save('keras.fall.model')
    score = model_single.evaluate(X_test,Y_test,verbose = 1)
    
    print('loss:',score[0])
    print('accuracy',score[1])
    
    y_pred_test_single = model_single.predict(X_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test_single = np.argmax(y_pred_test_single, axis=1)

    show_confusion_matrix(Y_test_gd, max_y_pred_test_single)

    single_accuracy.append(accuracy_score(Y_test_gd, max_y_pred_test_single))
    single_recall.append(recall_score(Y_test_gd, max_y_pred_test_single))
    single_precision.append(precision_score(Y_test_gd, max_y_pred_test_single))
    single_f1score.append(f1_score(Y_test_gd, max_y_pred_test_single))
    
    
    'multitask'
    input_layer = Input(shape = ((window_size-1),3,1),name='main_input')
    conv_1 = Conv2D(32,(3,3))(input_layer)
    maxpool_1 = MaxPooling2D(pool_size= (2,1))(conv_1)
    
    conv_2 = Conv2D(32,(3,1))(maxpool_1)
    max_pool_2 = MaxPooling2D(pool_size= (2,1))(conv_2)
    flatten_1 = Flatten()(max_pool_2)
    dense_1_1 = Dense(num_dense_unit)(flatten_1)
    activate_1_1 = Activation("relu")(dense_1_1)
    'branch 1'
    dense_1_2 = Dense(num_dense_unit)(activate_1_1)
    activate_1_2 = Activation("relu")(dense_1_2)
    dense_1_3 = Dense(num_dense_unit)(activate_1_2)
    activate_1_3 = Activation("relu")(dense_1_3)
    dense_1_4 = Dense(2)(activate_1_3)
    output_1 = Activation("softmax", name = "simple_output")(dense_1_4)
    
    'branch 2'
    dense_2_2 = Dense(num_dense_unit)(activate_1_1)
    activate_2_2 = Activation("relu")(dense_2_2)
    dense_2_3 = Dense(num_dense_unit)(activate_2_2)
    activate_2_3 = Activation("relu")(dense_2_3)
    dense_2_4 = Dense(80)(activate_2_3)
    output_2 = Activation("softmax", name = "subject_output")(dense_2_4)
    
    model_multi = Model(inputs = input_layer,outputs=[output_1, output_2])
    
    #print(model_multi.summary())
    
    'train model'
    
    model_multi =  Model(inputs=input_layer,outputs=[output_1,output_2])

    model_multi.compile(optimizer = 'sgd',
              loss = {'simple_output':'binary_crossentropy',
                      'subject_output':'categorical_crossentropy'},
              loss_weights = {'simple_output':alpha_per,
                              'subject_output':1.0 - alpha_per},
              metrics = ['accuracy']
              )

    model_multi.fit({'main_input':X_train},
          {'simple_output':Y_train,
           'subject_output':Z_train},
          epochs=20, batch_size=50)
    
    multi_score = model_multi.evaluate(X_test,[Y_test,Z_test],verbose = 1)
    
    
    #model_multi.save('keras.fall.model')
    print('total_loss:',multi_score[0])
    print('simple_loss',multi_score[1])
    print('complex_loss',multi_score[2])
    print('simple_accuracy',multi_score[3])
    print('complex_accuracy',multi_score[4])
    

    [y_pred_test_multi,z_pred_test_multi] = model_multi.predict(X_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test_multi = np.argmax(y_pred_test_multi, axis=1)
    max_z_pred_test_multi = np.argmax(z_pred_test_multi, axis=1)
    
    show_confusion_matrix(Y_test_gd, max_y_pred_test_multi)
    
    simple_accuracy.append(accuracy_score(Y_test_gd, max_y_pred_test_multi))
    simple_recall.append(recall_score(Y_test_gd, max_y_pred_test_multi))
    simple_precision.append(precision_score(Y_test_gd, max_y_pred_test_multi))
    simple_f1score.append(f1_score(Y_test_gd, max_y_pred_test_multi))
    complex_accuracy.append(accuracy_score(Z_test_gd, max_z_pred_test_multi))
    complex_cm.append(confusion_matrix(Z_test_gd, max_z_pred_test_multi, labels = class_list))
        
    
    
    
    
overall_simple_accuracy = sum(simple_accuracy)/len(simple_accuracy)
overall_simple_recall = sum(simple_recall)/len(simple_recall)
overall_simple_precision = sum(simple_precision)/len(simple_precision)
overall_simple_f1score = sum(simple_f1score)/len(simple_f1score)

overall_complex_accuracy = sum(complex_accuracy)/len(complex_accuracy)
overall_cmplex_cm = sum(complex_cm)

[ttest_acc, pvalue_acc] = ttest_rel(single_accuracy,simple_accuracy)
[ttest_pre, pvalue_pre] = ttest_rel(single_precision,simple_precision)
[ttest_rec, pvalue_rec] = ttest_rel(single_recall,simple_recall)
[ttest_f1, pvalue_f1] = ttest_rel(single_f1score,simple_f1score)


plot_model(model_single, to_file='single_outputs.png')
plot_model(model_multi, to_file='multiple_outputs.png')

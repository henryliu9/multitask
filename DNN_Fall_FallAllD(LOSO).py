from __future__ import print_function
from matplotlib import pyplot as plt
#matplotlib inlin
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML
import scipy.io as sio 


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import random

import pdb
import math
import h5py as h5

def SlidingWindow_ADL(D_Acc,label,WS):
    WindowList=np.zeros((1,WS,3),dtype=np.float32)
    WindowList[0,:,:]=D_Acc[0:WS]
    
    labels= label
    
    i=WS
    j=1
    while i+WS<=len(D_Acc):
        Window=np.zeros((1,WS,3),dtype=np.float32)
        Window[0,:,:]=D_Acc[i:i+WS]
        WindowList=np.vstack([WindowList,Window])
        j=j+1
        i=i+round(WS*0.5)
        labels = np.vstack([labels,label])   
    return WindowList, labels    

def SlidingWindow_Fall(D_Acc,label,WS,IP):
    WindowList=np.zeros((1,WS,3),dtype=np.float32)
    WindowList[0,:,:]=D_Acc[0:WS]
    
    labels= label
    
    i=WS
    j=1
    while i+WS<=len(D_Acc) & i<IP & i+WS >IP :
        Window=np.zeros((1,WS,3),dtype=np.float32)
        Window[0,:,:]=D_Acc[i:i+WS]
        WindowList=np.vstack([WindowList,Window])
        j=j+1
        i=i+round(WS*0.5)
        labels = np.vstack([labels,label])   
    return WindowList, labels 

def create_segments_and_labels(path,DataNameList,ImpactList,WS):
    DataNameList=DataNameList.reset_index()
    Total_WindowList=np.empty((0,WS,3))
    Total_labels=np.empty(0)
    j=0;
    for i in DataNameList["DataName"]:
        label=DataNameList["BinaryLabel"][j]
        filename=path+i
        C_df=pd.read_csv(filename,header=None)        
        D_Acc=C_df[:].values[:].astype('float32')
        
        if DataNameList["BinaryLabel"][j]==1:
            filter=(ImpactList["DataName"]==i)
            IP=ImpactList[filter]
            C_WindowList, C_labels =SlidingWindow_Fall(D_Acc,label,WS,IP)
        else:
            C_WindowList, C_labels =SlidingWindow_ADL(D_Acc,label, WS)
        Total_WindowList=np.append(Total_WindowList,C_WindowList )        
        Total_labels=np.append(Total_labels,C_labels )
            
        j=j+1
        
# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)
# Same labels will be reused throughout the program
LABELS = ['ADL','Fall']


FS=238 #FallAllD
WS=FS*7
DataSet_Name='FallAllD'
Position='ankle'
# path='/Users/kai-chunliu/Documents/code/Fall Detection/Data/SisFall_dataset/ImpactWindow(resample)/waist_25Hz/'
path1='/Users/kai-chunliu/Documents/code/Fall Detection/Data/FallAllD/data_management/'
df=pd.read_csv(path1+'FallAllD_DataNameList.csv',)

#dataNameList load
filter=(df["Position"]==2)
DataNameList=df[filter]
for i in [12,11,10,4]:
    filter2=(DataNameList["Subject"]!=i)
    DataNameList=DataNameList[filter2]

#ImpactList load
path2='/Users/kai-chunliu/Documents/code/Fall Detection/Data/FallAllD/ImpactPoint_Check/'
ImpactList=pd.read_csv(path2+'IP_table_wrist.csv',)

#LOSO
SubjectList=pd.unique(pd.Series(DataNameList["Subject"]))
for i in SubjectList:
    filter=(DataNameList["Subject"]!=i)
    DataNameList_train=DataNameList[filter]
    filter=(DataNameList["Subject"]==i)
    DataNameList_test=DataNameList[filter]
    
    xtrain, ytrain = create_segments_and_labels(path1,DataNameList_train,ImpactList,FS)




W1,W2=2, 1.5
TIME_PERIODS=round(W1*FS)+round(W2*FS)+1

# Hyper-parameters
BATCH_SIZE = 10000
EPOCHS = 100


y_test_all=np.array([])
y_pred_test_all=np.array([])

# Label_List1=df[5].values[1:]#fall and ADL
# Label_List2=df[2].values[1:] #detail fall and ADL types


def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #plt.savefig('Confusion.png', dpi=150)
    plt.show()

def create_segments_and_labels(path,df,W1,W2,FS):
    DataList=df[0].values[:]
    labels=df[5].values[:]#fall and ADL

    #check the segment length by reading the first segment
    filename=path+DataList[0]
    C_df=pd.read_csv(filename,header=None)     
    
    Window_list=np.zeros((len(DataList),len(C_df),3),dtype=np.float32)
    
    j=0
    for i in DataList[:]:
        filename=path+i
                         
        C_df=pd.read_csv(filename,header=None)        
        D_Acc=C_df[:].values[:].astype('float32')
        Window_list[j,:,:]=D_Acc
        j=j+1

    labels = np.asarray(labels)
    
    return Window_list, labels

#leave one subject out cross valudation
df_r=df[1:]

#list the unique subject index
subject_ind=df_r[1].unique()
subject_ind= [str(i) for i in subject_ind]


for testInd in subject_ind:
    df_test = df_r.loc[df_r[1]==testInd]
    df_train = df_r.loc[df_r[1]!=testInd]
    
    # Load data set containing all the data from csv
    x_train, y_train = create_segments_and_labels(path,df_train,W1,W2,FS)
    x_test, y_test = create_segments_and_labels(path,df_test,W1,W2,FS)

    # Set input & output dimensions
    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    num_classes = 2    

    input_shape = (num_time_periods*num_sensors)
    x_train = x_train.reshape(x_train.shape[0], input_shape)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    print('New y_train shape: ', y_train.shape)
    y_train_hot = np_utils.to_categorical(y_train, num_classes)

    x_test = x_test.reshape(x_test.shape[0], input_shape)
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    print('New y_test shape: ', y_test.shape)
    y_test_hot = np_utils.to_categorical(y_test, num_classes)


    print('Model built')
    model_m = Sequential()
    # Remark: since coreml cannot accept vector shapes of complex shape like
    # [80,3] this workaround is used in order to reshape the vector internally
    # prior feeding it into the network
    model_m.add(Reshape((num_time_periods, 3), input_shape=(input_shape,)))
    print(model_m.summary())
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Flatten())
    model_m.add(Dense(num_classes, activation='softmax'))
    print(model_m.summary())

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model_'+DataSet_Name+'_'+Position+'_'+testInd+'.h5', #best_model.{epoch:02d}-{val_loss:.2f}.h5
            monitor='val_loss', save_best_only=True,mode='min') # filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        ]

    model_m.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])



    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model_m.fit(x_train,
                          y_train_hot,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks_list,
                          validation_split=0.2,
                          verbose=1)
    
    # plt.figure(figsize=(6, 4))
    # plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
    # plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
    # plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    # plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    # plt.title('Model Accuracy and Loss')
    # plt.ylabel('Accuracy and Loss')
    # plt.xlabel('Training Epoch')
    # plt.ylim(0)
    # plt.legend()
    # plt.savefig('Learning_curve_DDAE_MSE_'+DataSet_Name+'_'+Position+'_'+testInd+'.png', dpi=150)
    # plt.show()
    

    # Print confusion matrix for training data
    y_pred_train = model_m.predict(x_train)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    # print(classification_report(y_train, max_y_pred_train))




    y_pred_test = model_m.predict(x_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    # max_y_test = np.argmax(y_test, axis=1)

    # show_confusion_matrix(y_test, max_y_pred_test)

    print(classification_report(y_test, max_y_pred_test))
    y_test_all=np.append(y_test_all,y_test)
    y_pred_test_all=np.append(y_pred_test_all,max_y_pred_test)

print(classification_report(y_test_all, y_pred_test_all)) 
show_confusion_matrix(y_test_all, y_pred_test_all)   

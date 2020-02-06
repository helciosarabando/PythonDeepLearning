# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:57:25 2019

@author: hsarabando

Improving DL and ML Models for TEP prognostics 
(LSTM, Random Forest and XgBoost)

"""
#-----------------Importing all the necessary Packages------------------------
print ("\nImporting all the necessary packages and solving the dependencies\n")
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (multilabel_confusion_matrix,
                             confusion_matrix,
                             classification_report)
from sklearn.preprocessing import (StandardScaler,
                                   Normalizer)
from sklearn.model_selection import (RandomizedSearchCV,
                                     GridSearchCV,
                                     KFold,
                                     train_test_split)
from sklearn.ensemble import RandomForestClassifier

from keras.layers import (LSTM, 
                          BatchNormalization,
                          concatenate,
                          Flatten,
                          MaxPool2D,
                          Embedding,
                          Dense,
                          Dropout,
                          MaxPooling2D,
                          Reshape)
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import (Callback, 
                             EarlyStopping)

import xgboost as xgb

from prettytable import PrettyTable

import warnings
warnings.filterwarnings("ignore")
#---------------------------------------------------------------------------

repeat_exec = input("Is this the first execution of the algorithm? [y/n]: ")

if repeat_exec == 'y':
    M1_val_acc = 0
    M2_val_acc = 0
    M3_val_acc = 0
    M4_val_acc = 0
    M5_val_acc = 0
    M6_val_acc = 0

else:
    if repeat_exec == 'n':
        print("\nPerforming another execution...")
    
    else:
        print("\nWrong choice... Pay attention! - Initializing val_acc")
        M1_val_acc = 0
        M2_val_acc = 0
        M3_val_acc = 0
        M4_val_acc = 0
        M5_val_acc = 0
        M6_val_acc = 0

#-----------------------Selecting the Model to execute----------------------
print("\n")
t_select = PrettyTable()
t_select.field_names = ['Model - to be chosen', 'Selection Number']
t_select.add_row(['Model (LSTM) - 1', 1])
t_select.add_row(['Model (LSTM) - 2', 2])
t_select.add_row(['Model (LSTM) - 3', 3])
t_select.add_row(['Model (LSTM) - 4', 4])
t_select.add_row(['Random Forest', 5])
t_select.add_row(['XgBoost', 6])
print(t_select)


m_select = input("Select the Model to execute: ")
if m_select == "1":
    print ("\nYou chose Model (LSTM) - 1\n")
    model_num = 1
elif m_select == "2":
    print ("\nYou chose Model (LSTM) - 2\n")
    model_num = 2
elif m_select == "3":
    print ("\nYou chose Model (LSTM) - 3\n")
    model_num = 3
elif m_select == "4":
    print ("\nYou chose Model (LSTM) - 4\n")
    model_num = 4
elif m_select == "5":
    print ("\nYou chose Random Forest\n")
    model_num = 5
elif m_select == "6":
    print ("\nYou chose XgBoost\n")
    model_num = 6
    
else:
    print ("\nWrong choice, try againt! Pay attention on the table!\n")
    model_num = 7
#--------------------------------------------------------------------------
          
# Continuous flow without a "7" as a number of model chosen        
if model_num != 7:
    
#-------------Reading data from the Preprocessed Dataset-------------------
    print ("\nReading dataframes from the Preprocessed R_Dataset (TEP)\n")
# Reading the train, test and CV data
    train  = pd.read_csv("DATA/train.csv")
    cv     = pd.read_csv("DATA/cv.csv")
    test   = pd.read_csv("DATA/test.csv")

    train.head()

# Sorting the Datasets wrt (with respect to) to the simulation runs
    train.sort_values(['simulationRun','faultNumber'],inplace = True)
    test.sort_values(['simulationRun','faultNumber'],inplace = True)
    cv.sort_values(['simulationRun','faultNumber'],inplace = True)



#----------Performance Metric - FDR Fault Detection Rate--------------------
# Method to print the Confusion Matrix
    def plot_confusion_matrix(test_y, predict_y):
        C = confusion_matrix(test_y, predict_y)
        print("Number of misclassified points ",
              (len(test_y)-np.trace(C))/len(test_y)*100)
        # C = 9,9 matrix, each cell (i,j) represents number of points of class 
        # i are predicted class j
    
        A =(((C.T)/(C.sum(axis=1))).T)
        #divid each element of the confusion matrix with the sum of elements 
        #in that column
    
        # C = [[1, 2],
        #     [3, 4]]
        # C.T = [[1, 3],
        #        [2, 4]]
        # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds 
        # to rows in two diamensional array
        # C.sum(axix =1) = [[3, 7]]
        # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
        #                           [2/3, 4/7]]

        # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
        #                           [3/7, 4/7]]
        # sum of row elements = 1
    
        B =(C/C.sum(axis=0))
        #divid each element of the confusion matrix with the sum of elements 
        # in that row
        # C = [[1, 2],
        #     [3, 4]]
        # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds 
        # to rows in two diamensional array
        # C.sum(axix =0) = [[4, 6]]
        # (C/C.sum(axis=0)) = [[1/4, 2/6],
        #                      [3/4, 4/6]] 
    
        labels = [0,1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20]
        cmap=sns.light_palette("green")
        # representing A in heatmap format
        print("-"*50, "Confusion matrix", "-"*50)
        plt.figure(figsize=(20,20))
        sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", 
                xticklabels=labels, 
                yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
    
        print("-"*50, "Precision matrix", "-"*50)
        plt.figure(figsize=(20,20))
        sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", 
                    xticklabels=labels, 
                    yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        print("Sum of columns in precision matrix",B.sum(axis=0))
        
        # representing B in heatmap format
        print("-"*50, "Recall matrix"    , "-"*50)
        plt.figure(figsize=(20,20))
        sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", 
                    xticklabels=labels, 
                    yticklabels=labels)
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        print("Sum of rows in precision matrix",A.sum(axis=1))
    
# Removing faults 3,9 and 15
    print ("\nRemoving faults 3, 9 and 15\n")
    tr = train.drop(train[(train.faultNumber == 3) | (train.faultNumber == 9)\
                          | (train.faultNumber == 15)].index).reset_index()
    ts = test.drop(test[(test.faultNumber == 3) | (test.faultNumber == 9)\
                        | (test.faultNumber == 15)].index).reset_index()
    cv_ = cv.drop(cv[(cv.faultNumber == 3) | (cv.faultNumber == 9)\
                     | (cv.faultNumber == 15)].index).reset_index()
    
# Converting the class labels to categorical values and removing unnecessary 
# features from train, test and cv data.
    if model_num == 1 or model_num == 2:
        print ("\nConverting the class labels to categorical values\n")
        y_train = to_categorical(tr['faultNumber'],num_classes=21)
        y_test  = to_categorical(ts['faultNumber'],num_classes=21)
        y_cv    = to_categorical(cv_['faultNumber'],num_classes=21)

        tr.drop(['faultNumber','Unnamed: 0','simulationRun','sample','index'],
                axis=1,inplace=True)
        ts.drop(['faultNumber','Unnamed: 0','simulationRun','sample','index'],
                axis =1,inplace=True)
        cv_.drop(['faultNumber','Unnamed: 0','simulationRun','sample','index'],
                 axis =1,inplace=True)

# Resizing the train, test and cv data.
        print ("\nResizing the data (train, test and cv)\n")
        x_train = np.resize(tr,(235840,52,1)) # Esta matriz foi modificada de "183200"
        x_test  = np.resize(ts,(89000,52,1))
        x_cv    = np.resize(cv_,(93440,52,1))
#----------------------------------------------------------------------------

#-----------------Deep Learning Models---------------------------------------

    if model_num == 1: 
#________________________Model 1 - LSTM______________________________________
        print ("\nInitiating the construction of the model\n")
        model = Sequential()

# Constructing the model
        model.add(LSTM(256,input_shape= (52,1),return_sequences= True))
        model.add(LSTM(128,return_sequences= False))
        model.add(Dense(300))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Dense(21, activation='softmax'))

# Compiling the model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        print(model.summary()) 

# Training the model
        model.fit(x_train, y_train, 
                  nb_epoch=50,
                  verbose=1, 
                  batch_size=256, 
                  validation_data = (x_cv,y_cv))

# Printing the results
        model_paras = model.history
        x = list(range(1,len(model_paras.history['loss']) + 1))
        plt.figure(figsize=(5,5))
        plt.plot(x,model_paras.history['val_acc'],color = 'r',label = 'Test accuracy')
        plt.plot(x,model_paras.history['acc'],color = 'b',label = 'Training accuracy')
        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.show()

# Object to show results
        M1_val_acc = model_paras.history['val_acc'][-1]

# Final evaluation of the model
        score, acc = model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy:', acc)
        print("Test loss:", score)

# Printing the confusion matrix
        y = model.predict_proba(x_test)
        plot_confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(y,axis = 1))


    elif model_num == 2:
#________________________Model 2 - LSTM______________________________________
        print ("\nInitiating the construction of the model\n")
        model = Sequential()

# Constructing the model
        model.add(LSTM(128,input_shape = (52,1),return_sequences=False))
        model.add(Dense(300))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Dense(21,activation='softmax'))

# Compiling the model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])

        print(model.summary())

# Using EarlyStopping method to stop the train in the "maximum accuracy"
        es = EarlyStopping(monitor='val_acc', 
                           min_delta=10e-6, 
                           patience=3,
                           verbose=1,
                           mode='max',
                           baseline=None,
                           restore_best_weights=True)

# Training the model
        model.fit(x_train, y_train, 
                  nb_epoch=50,
                  verbose=1,
                  batch_size=256,
                  validation_data = (x_cv,y_cv),
                  callbacks=[es])

# Printing the results
        model_paras = model.history
        x = list(range(1,len(model_paras.history['loss']) + 1))
        plt.figure(figsize=(5,5))
        plt.plot(x,model_paras.history['val_acc'],color = 'r',label = 'Test loss')
        plt.plot(x,model_paras.history['acc'],color = 'b',label = 'Training loss')
        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.show()

# Object to show results
        M2_val_acc = model_paras.history['val_acc'][-1]

# Final evaluation of the model
        score, acc = model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy:', acc)
        print("Test loss:", score)

# Printing the confusion matrix
        y = model.predict_proba(x_test)
        plot_confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(y,axis = 1))


    elif model_num == 3:
#________________________Model 3 - LSTM______________________________________
# Fault is introduced at one hour of each simulation
# Train the LSTM models with 20 samples at a time
# Each input to the model will have 20 samples of 52 Dim from the train dataset

# Transforming the class labels of train, test, cv. Every 20th point is sampled.
        print ("\nTransforming the class labels for model 3\n")
        label_train = []

        for row in range(9160):
           
            label_train.append(tr['faultNumber'][row * 20])
        
            
        label_test = []
        for row in range(4450):
           
            label_test.append(ts['faultNumber'][row * 20])
            
        label_cv = []
        for row in range(4672):
            label_cv.append(cv['faultNumber'][row * 20])

# Transforming the train, test, cv data, to get an array of shape (20,52)
        tr_ = tr.values
        tr_arr = np.empty([9160,20,57])
        for x in range(9160):
            tr_arr[x] = tr_[20 * x : 20+20*x]


        ts_ = ts.values
        ts_arr = np.empty([4450,20,57])
        
        for x in range(4450):
            ts_arr[x] = ts_[20 * x : 20+20*x]


        cv_ar = cv_.values
        cv_arr = np.empty([4672,20,57])
        for x in range(4672):
            cv_arr[x] = cv_ar[20 * x : 20+20*x]
            
        y_train  = to_categorical(label_train,num_classes=21)
        y_test   = to_categorical(label_test,num_classes=21)
        y_cv     = to_categorical(label_cv,num_classes=21)
        
        x_train  = np.transpose(tr_arr,(0,2,1))
        x_test   = np.transpose(ts_arr,(0,2,1))
        x_cv     = np.transpose(cv_arr,(0,2,1))

# Initiating the model
        print ("\nInitiating the construction of the model\n")
        model = Sequential()

# Constructing the model
        model.add(LSTM(128,input_shape=(57, 20),return_sequences= True,activation = 'sigmoid'))
        model.add(LSTM(128,return_sequences= False,activation = 'sigmoid'))
        model.add(Dense(300))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Dropout(0.8))
        model.add(BatchNormalization())
        model.add(Dense(21,activation='softmax'))

# Compiling the model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])

        print(model.summary())

# Training the model
        model.fit(x_train, y_train, 
                  nb_epoch=50,
                  verbose=1,
                  batch_size=256,
                  validation_data = (x_test,y_test))

# Printing the results
        model_paras = model.history
        x = list(range(1,len(model_paras.history['loss']) + 1))
        plt.figure(figsize=(5,5))
        plt.plot(x,model_paras.history['val_acc'],color = 'r',label = 'Test accuracy')
        plt.plot(x,model_paras.history['acc'],color = 'b',label = 'Training accuracy')
        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.show()

# Object to show results
        M3_val_acc = model_paras.history['val_acc'][-1]

# Final evaluation of the model
        score, acc = model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy:', acc)
        print("Test loss:", score)

# Printing the confusion matrix
        y = model.predict_proba(x_test)
        plot_confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(y,axis = 1))


    elif model_num == 4:
#________________________Model 4 - LSTM______________________________________
# Fault is introduced at one hour of each simulation
# Train the LSTM models with 20 samples at a time
# Each input to the model will have 20 samples of 52 Dim from the train dataset
# Same data preparation that was done in model 3

        print ("\nInitiating the construction of the model\n")
        model = Sequential()

# Constructing the model
        model.add(LSTM(128,input_shape= (52,20),return_sequences= True,activation = 'sigmoid'))
        model.add(LSTM(128,return_sequences= False,activation = 'sigmoid'))
        model.add(Dense(300))
        model.add(Dropout(0.4))
        model.add(Dense(21,activation='softmax'))

# Compiling the model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])

        print(model.summary())

# Training the model
        model.fit(x_train, y_train, 
                  nb_epoch=100,
                  verbose=1,
                  batch_size=256,
                  validation_data = (x_test,y_test))

# Printing the results
        model_paras = model.history
        x = list(range(1,len(model_paras.history['loss']) + 1))
        plt.figure(figsize=(5,5))
        plt.plot(x,model_paras.history['val_acc'],color = 'r',label = 'Test accuracy')
        plt.plot(x,model_paras.history['acc'],color = 'b',label = 'Training accuracy')
        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.show()

# Object to show results
        M4_val_acc = model_paras.history['val_acc'][-1]

# Final evaluation of the model
        score, acc = model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy:', acc)
        print("Test loss:", score)

# Printing the confusion matrix
        y = model.predict_proba(x_test)
        plot_confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(y,axis = 1))


    elif model_num == 5 or model_num == 6:
#_______Models 5 e 6 - Classical ML models (Random Forest and XgBoost)________
# Starting with Random Forest

# Converting the class labels to categorical values and removing unnecessary 
# features from train, test and cv data.
        print ("\nPreprocessing the Dataset for M.L. Methods\n")
        y_train = tr['faultNumber']
        y_test = ts['faultNumber']
        y_cv = cv_['faultNumber']
        tr.drop(['faultNumber','Unnamed: 0','simulationRun','sample','index'],
                axis=1,inplace=True)
        ts.drop(['faultNumber','Unnamed: 0','simulationRun','sample','index'],
                axis =1,inplace=True)
        cv_.drop(['faultNumber','Unnamed: 0','simulationRun','sample','index'],
                 axis =1,inplace=True)

        standard_scalar  = StandardScaler()
        train_norm       = standard_scalar.fit_transform(tr)
        test_norm        = standard_scalar.transform(ts)
        cv_norm          = standard_scalar.transform(cv_)

# Model fitting and hyperparameter tuning using gridsearch
        if model_num == 5:
            print ("\nInitiating the construction of the model\n")
            x_cfl=RandomForestClassifier()
    
            prams={
                 'n_estimators':[100,200,500],
                 'max_depth':[15,20,25,30,35]
       
            }
            model=GridSearchCV(x_cfl,
                               param_grid=prams,
                               verbose=10,
                               n_jobs=-1,
                               scoring='f1_micro',
                               cv=3)
            model.fit(train_norm,y_train)
            print("\nBest estimator is", model.best_params_)

# Training the model
            clf = RandomForestClassifier(n_jobs=-1,
                                         verbose=1,
                                         n_estimators= 500,
                                         max_depth=35)
            clf.fit(train_norm, y_train)
            print(clf.score(train_norm, y_train))

# Printing the confusion matrix
            plot_confusion_matrix(y_test, clf.predict(test_norm))
            
            M5_val_acc = 0.8986

#**********FDR for Random Forest = 0.8986 ???

# Now trying XgBoost
        if model_num == 6:
            print ("\nInitiating the construction of the model\n")
# Model fitting and hyperparameter tunning using gridsearch
            x_cfl = xgb.XGBClassifier(objective="multi:softprob")
    
            prams={
                
                 'n_estimators':[100,200,500],
                 'max_depth':[5,10,15,20,30,35]
            }
            model=GridSearchCV(x_cfl,
                               param_grid=prams,
                               verbose=10,
                               n_jobs=-1,
                               scoring='f1_micro',
                               cv=3)
            model.fit(train_norm,y_train)
            
            clf=xgb.XGBClassifier(objective="multi:softprob",
                                  n_estimators=500,
                                  max_depth=20,
                                  n_jobs=-1)
            clf.fit(train_norm, y_train)

# Printing the confusion matrix
            y_pred_test = clf.predict(test_norm)

#**********FDR for XgBoost = 0.9288 ???

            plot_confusion_matrix(y_test, y_pred_test)
            print("Best estimator is", model.best_params_)
    
            M6_val_acc = 0.9288
            
#__________________________Results______________________________

    print ("\nShowing a table for all models results\n")
    table = PrettyTable()
    table.field_names = ['Model - D.L. & M.L.', 'Average FDR']
    table.add_row(['Model (LSTM) - 1', round(M1_val_acc,4)])
    table.add_row(['Model (LSTM) - 2', round(M2_val_acc,4)])
    table.add_row(['Model (LSTM) - 3', round(M3_val_acc,4)])
    table.add_row(['Model (LSTM) - 4', round(M4_val_acc,4)])
    table.add_row(['Random Forest', round(M5_val_acc,4)])
    table.add_row(['XgBoost', round(M6_val_acc,4)])
    print(table)

else:
    print ("\nNothing has done")
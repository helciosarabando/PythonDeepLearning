# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:27:39 2020

@author: hsarabando

Improving "Deep Learning" Algorithms for TEP prognostics
(Tennessee Eastman Process)
This Algorithm is a RNN-based model called LSTM 
(Long Short-term Memory)
This will be called "LSTM - model 1"

"""
print (__doc__)

#-----------------Importing all the necessary Packages------------------------
print ("\nImporting all the necessary packages and solving the dependencies\n")
import pandas as pd
print("Pandas:{}".format(pd.__version__))

import seaborn as sns
print("Seaborn:{}".format(sns.__version__))

import numpy as np
np.random.seed(123)
print("NumPy:{}".format(np.__version__))

import matplotlib.pyplot as plt
import matplotlib as mpl
print("Matplotlib:{}".format(mpl.__version__))

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
import sklearn as sk
print("sklearn:{}".format(sk.__version__))

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
import keras
print("Keras:{}".format(keras.__version__))

from prettytable import PrettyTable


#---------------------- Just for Warnings -----------------------
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
#----------------------------------------------------------------

#----------------------------------------------------------------
#-------------------------- Header ------------------------------
#----------------------------------------------------------------
print ('\n')
M1_val_acc = 0
t_select = PrettyTable()
t_select.field_names = ['Model', 'Description']
t_select.add_row(['LSTM - 1', '2 LSTM layers, 1 Dense layer, Dropout, 2 Dense layers'])
print(t_select)

#-------------Reading data from the Preprocessed Dataset-------------------
print ("\nReading dataframes from the Preprocessed R_Dataset (TEP)\n")
# Reading the train, test and CV data
train  = pd.read_csv("DATA/train.csv")
cv     = pd.read_csv("DATA/cv.csv")
test   = pd.read_csv("DATA/test.csv")

# Sorting the Datasets wrt (with respect to) to the simulation runs
train.sort_values(['simulationRun','faultNumber'],inplace = True)
test.sort_values(['simulationRun','faultNumber'],inplace = True)
cv.sort_values(['simulationRun','faultNumber'],inplace = True)

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
#----------------------------------------------------------------------------

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
                  epochs=50,
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

#__________________________Result______________________________
print ("\nShowing a table for all models results\n")
table = PrettyTable()
table.field_names = ['Model', 'Average FDR']
table.add_row(['LSTM - 1', round(M1_val_acc,4)])
print(table)
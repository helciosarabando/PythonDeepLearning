# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:57:25 2019

@author: hsarabando

Improving Methods to Preprocess the TE Process 

"""
#-----------------Importing all the necessary Packages------------------------
print ("\nImporting all the necessary packages and solving the dependencies\n")
import pandas as pd

import pyreadr as py # library to read .Rdata files in python

import matplotlib.pyplot as plt

import missingno as msno

import warnings
warnings.filterwarnings("ignore")
#--------------------------------------------------------------------------

#-----------Preprocessing the Dataset--------------------------------------
# Reading train data in .R format
print ("\nReading TRAIN data in .R format\n")
a1 = py.read_r("DATA\TEP_FaultFree_Training.RData")
a2 = py.read_r("DATA\TEP_Faulty_Training.RData")

# Reading test data in .R format
print ("\nReading TEST data in .R format\n")
a3 = py.read_r("DATA\TEP_FaultFree_Testing.RData")
a4 = py.read_r("DATA\TEP_Faulty_Testing.RData")

# Printing the objects recently updated
print("\nObjects that are present in a1 :",a1.keys())
print("\nObjects that are present in a2 :",a2.keys())
print("\nObjects that are present in a3 :",a3.keys())
print("\nObjects that are present in a4 :",a4.keys())

# Reading the .Rdata files in "pandas dataframe" and saving it in .csv file
print ("\n\nConverting .Rdata in pandas dataframe and saving in .csv files\n")
# Reading train data
print ("\nConverting TRAIN data\n")
b1 = a1['fault_free_training']
b2 = a2['faulty_training']
# Reading test data
print ("\nConverting TEST data\n")
b3 = a3['fault_free_testing']
b4 = a4['faulty_testing']

# Concatinating the "train" and the "test" dataset
print ("\nConcatinating the TRAIN and the TEST dataset\n")
frames_tr = [b1,b2]
train_ts = pd.concat(frames_tr)
frames_ts = [b3,b4]
test = pd.concat(frames_ts)

#--------------Visualizing properties of the Dataset--------------------------
# Visualizing the dataset shape and distributions 
print("\nShape of the TRAIN dataset:", train_ts.shape)
print("\nShape of the TEST dataset:", test.shape)

print("\nDistribution of faults in TRAIN dataset:")
print(train_ts['faultNumber'].value_counts())

print("\nDistribution of faults in TEST dataset:")
print(test['faultNumber'].value_counts())

# Description of the data in Dataset
train_ts.describe() # É possível colocar o resultado em arquivo?!?

# Plot to indicate the number of "missing values" in each column of the dataset.
columns_names = train_ts.columns
msno.bar(train_ts[columns_names[3:55]])

h1_select = input("\nWould you like to plot the histograms of the features in TRAIN dataset? [y/n]: ")
if h1_select == 'y':
# Loop to plot the histogram of the features in the "Train" dataset
    for col in columns_names[3:]:
        plt.figure(figsize=(5,5))
        plt.hist(train_ts[col])
        plt.xlabel(col)
        plt.ylabel("counts")
        plt.show()
    
    
elif h1_select == 'n':
    print("\nHistograms not showed")

else:
    print("\nWrong choice, histograms not showed")
    

test.describe()
    
h2_select = input("\nWould you like to plot the histograms of the features in TEST dataset? [y/n]: ")
if h2_select == 'y':
# Loop to plot the histogram of the features in the "Test" dataset
    for col in columns_names[3:]:
        plt.figure(figsize=(5,5))
        plt.hist(test[col])
        plt.xlabel(col)
        plt.ylabel("counts")
        plt.show()

elif h2_select == 'n':
    print("\nHistograms not showed")

else:
    print("\nWrong choice, histograms not showed")

#----------------Data Preparation for Deep Learning Models--------------------
print("\nData Preparation for Deep Learning Models - Pandas Dataframes\n")
Sampled_train  = pd.DataFrame()    # dataframe to store the train dataset
Sampled_test   = pd.DataFrame()    # dataframe to store test 
Sampled_cv     = pd.DataFrame()    # dataframe to store cv data

# Program to construct the sample "Train data"
print("\nProgram to construct the sample TRAIN data\n")
frame = []
for i in set(train_ts['faultNumber']): 
    b_i = pd.DataFrame()
    if i == 0:
        b_i = train_ts[train_ts['faultNumber'] == i][0:20000]
        frame.append(b_i)
    else:
        fr = []
        b = train_ts[train_ts['faultNumber'] == i]
        for x in range(1,25):
            b_x = b[b['simulationRun'] == x][20:500]
            fr.append(b_x)
        
        b_i = pd.concat(fr)
        
    frame.append(b_i)      
Sampled_train = pd.concat(frame)

# Program to construct the sample "CV data"
print("\nProgram to construct the sample CV data\n")
frame = []
for i in set(train_ts['faultNumber']):
    b_i = pd.DataFrame()
    if i == 0:
        b_i = train_ts[train_ts['faultNumber'] == i][20000:30000]
        frame.append(b_i)
    else:
        fr = []
        b = train_ts[train_ts['faultNumber'] == i]
        for x in range(26,35):
            b_x = b[b['simulationRun'] == x][20:500]
            fr.append(b_x)
        
        b_i = pd.concat(fr)
        
    frame.append(b_i)      
Sampled_cv = pd.concat(frame)

# Program to construct sample "Test data"
print("\nProgram to construct the sample TEST data\n")
frame = []
for i in set(test['faultNumber']):
    b_i = pd.DataFrame()
    if i == 0:
        b_i = test[test['faultNumber'] == i][0:2000]
        frame.append(b_i)
    else:
        fr = []
        b = test[test['faultNumber'] == i]
        for x in range(1,11):
            b_x = b[b['simulationRun'] == x][160:660]
            fr.append(b_x)
        
        b_i = pd.concat(fr)
        
    frame.append(b_i)      
Sampled_test = pd.concat(frame)

# Storing the Train, Test and CV dataset into .csv file for further use.
print("\nStoring Dataframes into .csv files for further use\n")
Sampled_train.to_csv("DATA/train.csv")
Sampled_test.to_csv("DATA/test.csv")
Sampled_cv.to_csv("DATA/cv.csv")

train  = pd.read_csv("DATA/train.csv")
cv     = pd.read_csv("DATA/cv.csv")
test   = pd.read_csv("DATA/test.csv")

train.head()               # Summary of the train Dataset

# Visualizing the shape of the train, cv and test Dataset
print("\nShape of the sampled TRAIN data:", train.shape)
print("\nShape of the sampled TEST data:", test.shape)
print("\nShape of the sampled CV data:", cv.shape)
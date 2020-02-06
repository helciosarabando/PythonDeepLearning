# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:26:03 2019

@author: hsarabando

TensorFlow "RBM - Restricted Boltzmann Machine"
RBM - A shallow network that learn to reconstruct data by themselves in an 
unsupervised fashion. It can extract meaningful features from a given input
"""
print(__doc__)

# Loading Packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt

#------------------- RBM as a "toy example" using TEP --------------------
# Hyperparameters
VisibleUnits = 52
HiddenUnits = 10

# Defining bias of visible and hidden layers
v_bias = tf.compat.v1.placeholder("float", [VisibleUnits])
h_bias = tf.compat.v1.placeholder("float", [HiddenUnits])

# Defining W (weights), Tensor of 52x21 (52 visbles and 21 hidden layers)
W = tf.constant(np.random.normal(loc=0.0, 
                                 scale=1.0, 
                                 size=(VisibleUnits,
                                       HiddenUnits)).astype(np.float32))

#-------------- Phase 1 - Forward Pass ----------------------------------- 

sess = tf.compat.v1.Session()

# Preprocessing the "Inputs"
X_data = pd.read_csv('DATA/train.csv', nrows=5)
X_data = X_data.drop(['faultNumber',
                      'Unnamed: 0',
                      'simulationRun',
                      'sample'],
                     axis=1)
X_5lines = np.array(X_data)
scaler = MinMaxScaler().fit(X_5lines)
X_scale = scaler.transform(X_5lines)
X_1line = np.array(X_scale[2])
X = tf.constant(X_1line, dtype = tf.float32, shape=[1,52])
v_state = X
print ("\nInput: ", sess.run(v_state))

# Bias from the hidden layer
h_bias = tf.constant(np.random.rand(1,HiddenUnits), dtype = tf.float32)
print ("\nh_bias: ", sess.run(h_bias))
print ("\nweight: ", sess.run(W))

# Calculate the probabilities of turning the hidden units on:
# h_prob = probabilities of the hidden units
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)  
print ("\np(h|v): ", sess.run(h_prob))


# Draw samples from the distribution:
# h_state = states
h_state = tf.nn.relu(tf.sign(h_prob - tf.random.uniform(tf.shape(h_prob))))
print ("\nh0 states:", sess.run(h_state))

#-------------------------------------------------------------------------

#-------- Phase 2 - Backward Pass - Reconstruction of the inputs ---------

vb = tf.constant(np.random.rand(1,VisibleUnits), dtype = tf.float32)
print ("\nbias: ", sess.run(vb))
v_prob = sess.run(tf.nn.sigmoid(tf.matmul(h_state, 
                                          tf.transpose(W)) + vb))
print ("\np(vi∣h): ", v_prob)
v_state = tf.nn.relu(tf.sign(v_prob - \
                             tf.random_uniform(tf.shape(v_prob))))
print ("\nv probability states: ", sess.run(v_state))

# What is the probability of generating the real inputs based on...
# the probability distribution 
# How similar X and V tensors are?
inp = sess.run(X)
print("\n", inp)
print("\n", v_prob[0])
v_probability = 1
for elm, p in zip(inp[0],v_prob[0]) :
    if elm ==1:
        v_probability *= p
    else:
        v_probability *= (1-p)
print("\n", v_probability, "\n")
sess.close()



#************************************************************************
#-----------------------RBM Model 1--------------------------------------
#************************************************************************

#----------- Importing and Preprocessing Dataset ------------------------
# Importing training data
print ("\nImporting training Data\n")
trX_data = pd.read_csv("DATA/train.csv", verbose =1)

# Dropping faults numbers 3, 9 and 15
trX_frame = trX_data.drop(trX_data[(trX_data.faultNumber == 3) \
                               | (trX_data.faultNumber == 9) \
                               | (trX_data.faultNumber == 15)] \
                          .index).reset_index()
# Dropping colummns "faultNumber", "0", "simulationRun", ... 
# "sample" and "index"
trX_frame = trX_frame.drop(['faultNumber',
                            'Unnamed: 0',
                            'simulationRun',
                            'sample',
                            'index'],
                           axis=1)
# Converting the dataframe into Numpy array
trX_array = np.array(trX_frame)
scaler =  Normalizer().fit(trX_array)
trX_scaler = scaler.transform(trX_array)

# Converting a Numpy array into Tensor
trX = tf.constant(trX_scaler, dtype = tf.float32, shape =[235840, 52])

# Importing the targeting of training data
print ("\nImporting the targeting of training Data\n")
trY_data = pd.read_csv("DATA/cv.csv", verbose =1)
trY_frame = trY_data.drop(trY_data[(trY_data.faultNumber == 3) \
                               | (trY_data.faultNumber == 9) \
                               | (trY_data.faultNumber == 15)] \
                          .index).reset_index()
trY_frame = trY_frame.drop(['faultNumber',
                            'Unnamed: 0',
                            'simulationRun',
                            'sample',
                            'index'],
                           axis=1)
trY_array = np.array(trY_frame)
scaler = Normalizer().fit(trY_array)
trY_scaler = scaler.transform(trY_array)
trY = tf.constant(trY_scaler, dtype = tf.float32, shape =[106400, 52])

# Importing test data
print ("\nImporting test Data\n")
teX_data = pd.read_csv("DATA/test.csv", verbose =1)
teX_frame = teX_data.drop(teX_data[(teX_data.faultNumber == 3) \
                               | (teX_data.faultNumber == 9) \
                               | (teX_data.faultNumber == 15)] \
                          .index).reset_index()
teX_frame = teX_frame.drop(['faultNumber',
                            'Unnamed: 0',
                            'simulationRun',
                            'sample',
                            'index'],
                           axis=1)
teX_array = np.array(teX_frame)
scaler = Normalizer().fit(teX_array)
teX_scaler = scaler.transform(teX_array)
teX = tf.constant(teX_scaler, dtype = tf.float32, shape =[104000, 52])

# Importing the target of the test data
print ("\nImporting the targeting of test Data\n")
teY_data = pd.read_csv("DATA/cv.csv", verbose =1)
teY_frame = teY_data.drop(teY_data[(teY_data.faultNumber == 3) \
                               | (teY_data.faultNumber == 9) \
                               | (teY_data.faultNumber == 15)] \
                          .index).reset_index()
teY_frame = teY_frame.drop(['faultNumber',
                            'Unnamed: 0',
                            'simulationRun',
                            'sample',
                            'index'],
                           axis=1)
teY_array = np.array(teY_frame)
scaler = Normalizer().fit(teY_array)
teY_scaler = scaler.transform(teY_array)
teY = tf.constant(teY_scaler, dtype = tf.float32, shape =[106400, 52])

#-----------------------------------------------------------------------

# Defining the parameters of the model
# TEP have 52 sensors signals and 21 faults (hidden nodes)
vb = tf.compat.v1.placeholder("float", [VisibleUnits])
hb = tf.compat.v1.placeholder("float", [HiddenUnits])

# W is a Tensor of 52x21 (weight between the neurons)
W = tf.compat.v1.placeholder("float", [VisibleUnits, HiddenUnits])

# Define the visible layer
v0_state = tf.compat.v1.placeholder("float", [None, VisibleUnits])

# Define the hidden layer
# h0_prob = probabilities of the hidden units
h0_prob = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)  
# h0_state = sample h given X 
h0_state = tf.nn.relu(tf.sign(h0_prob - \
                              tf.random_uniform(tf.shape(h0_prob))))

# Define reconstruction Tensors
# v1_prob = probabilities of the visible reconstructed units
v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, 
                                  tf.transpose(W)) + vb) 
# v1_state = sample v given h
v1_state = tf.nn.relu(tf.sign(v1_prob - \
                              tf.random_uniform(tf.shape(v1_prob))))

# Calculate an error as a means of elements across dimensions of a Tensor
err = tf.reduce_mean(tf.square(v0_state - v1_state))

# Gibbs Sampling
h1_prob = tf.nn.sigmoid(tf.matmul(v1_state, W) + hb)
# h1_state = sample h given X
h1_state = tf.nn.relu(tf.sign(h1_prob - \
                              tf.random_uniform(tf.shape(h1_prob))))

# Contrastive Divergence (CD-k) assuming k = 1
alpha = 0.01
W_Delta = tf.matmul(tf.transpose(v0_state), 
                    h0_prob) - tf.matmul(tf.transpose(v1_state), 
                                         h1_prob)
update_w = W + alpha * W_Delta
update_vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
update_hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0)


# Start a session and initialize the variables
cur_w = np.zeros([VisibleUnits, HiddenUnits], np.float32)
cur_vb = np.zeros([VisibleUnits], np.float32)
cur_hb = np.zeros([HiddenUnits], np.float32)
prv_w = np.zeros([VisibleUnits, HiddenUnits], np.float32)
prv_vb = np.zeros([VisibleUnits], np.float32)
prv_hb = np.zeros([HiddenUnits], np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Lets look at the error of the first run
sess.run(err, 
         feed_dict={v0_state: trX_scaler, 
                    W: prv_w, 
                    vb: prv_vb, 
                    hb: prv_hb})

# HyperParameters
epochs = 10
batchsize = 100
weights = []
errors = []

print ("\n")
for epoch in range(epochs):
    for start, end in zip( range(0, len(trX_scaler), batchsize), 
                          range(batchsize, len(trX_scaler), batchsize)):
        batch = trX_scaler[start:end]
        cur_w = sess.run(update_w, feed_dict={ v0_state: batch, 
                                              W: prv_w, 
                                              vb: prv_vb, 
                                              hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0_state: batch, 
                                                W: prv_w, 
                                                vb: prv_vb, 
                                                hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={ v0_state: batch, 
                                                W: prv_w, 
                                                vb: prv_vb, 
                                                hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(err, feed_dict={v0_state: trX_scaler, 
                                                   W: cur_w, 
                                                   vb: cur_vb, 
                                                   hb: cur_hb}))
            weights.append(cur_w)
    print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()

# What is the weights after training
uw = weights[-1].T

# A weight matrix of shape (21,52)
print ("\nWeights:", uw)

## Feed the sample case into the network and reconstruct the output
print ("\nJust running the sample case")
sample_case = X_5lines
hh0_p = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)
hh0_s = tf.nn.relu(tf.sign(hh0_p - tf.random_uniform(tf.shape(hh0_p)))) 
hh0_s = tf.round(hh0_p)
hh0_p_val,hh0_s_val  = sess.run((hh0_p, hh0_s), 
                                feed_dict={ v0_state: sample_case, 
                                           W: prv_w, hb: prv_hb})
print("\nProbability nodes in hidden layer \
      (5 lines sample case):" ,hh0_p_val)
print("\nactivated nodes in hidden layer \
      (5 lines sample case):" ,hh0_s_val)

# Reconstruct
vv1_p = tf.nn.sigmoid(tf.matmul(hh0_s_val, tf.transpose(W)) + vb)
rec_prob = sess.run(vv1_p, feed_dict={ hh0_s: hh0_s_val, 
                                      W: prv_w, 
                                      vb: prv_vb})
print("\nProbability of the Reconstruction \
      (5 lines sample case):" ,rec_prob)
# -*- coding: utf-8 -*-
# import tensorflow as tf
import pandas as pd
import scipy.io as scio
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import ttest_rel
from scipy import stats
from scipy import interp

import random
import cPickle
import os
import math
import h5py
import time
from itertools import cycle
from collections import Counter

import imageio
from images2gif import writeGif
from PIL import Image

import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
# from tflearn.layers.normalization import local_response_normalization
# from tflearn.layers.merge_ops import merge
# from tflearn.layers.estimator import regression
from sklearn import svm,preprocessing
from sklearn import datasets,svm,feature_selection
from sklearn.metrics import *
from sklearn.datasets import make_classification,load_iris,make_multilabel_classification
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel,RFECV,VarianceThreshold,SelectKBest,SelectPercentile,chi2,f_classif,f_regression
from sklearn.model_selection import  StratifiedKFold,train_test_split,GridSearchCV,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, IncrementalPCA,SparsePCA,FastICA,MiniBatchSparsePCA
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_decomposition import CCA
from sklearn.tree import DecisionTreeClassifier

def load_data(train_data = True,one_hot=True):
    if train_data==1:
        data_path = 'data/train.xlsx'
        origin_excel = pd.read_excel(data_path, 'train', header=0)
    if train_data==0:
        data_path = 'data/test.xlsx'
        origin_excel = pd.read_excel(data_path, 'test', header=0)
    gender = origin_excel['GENDER']
    gender2num = np.zeros((len(gender),1),dtype='int32')
    for index,value in enumerate(gender):
        if value=='Female':
            gender2num[index] = 0
        else:
            gender2num[index] = 1
    if train_data == 1:
        label = origin_excel['Diagnosis']
        label2num = np.zeros((len(label),1),dtype='int32') # word label
        for index,value in enumerate(label):
            if value == 'cMCI':
                label2num[index] = 0
            elif value == 'AD':
                label2num[index] = 1
            elif value == 'HC':
                label2num[index] =2
            else:
                label2num[index] = 3
    data = origin_excel.values
    if train_data == 1:
        data = data[:,3:]  # keep data from "age" to "end"
    if train_data == 0:
        data = data[:,2:]
    data = np.hstack((gender2num,data))
    data_final = data.astype('float32')

    if train_data == 1:
        return data_final,label2num
    if train_data == 0:
        return data_final
def z_score(data):
    std = np.std(data,axis = 0)
    mean = np.mean(data,axis = 0)
    zscore_data = (data-mean)/std
    return zscore_data
def choose_perplexity_tsne():
    i = 0
    plt.figure(figsize=(12, 10))
    for perplexity in range(0,90,10):
        i +=1
        sub_plot = '3'+'3'+ str(i)
        plt.subplot(sub_plot)#axisbg='k'
        ts = TSNE(n_components=2, perplexity=perplexity,init='pca', random_state=0,n_iter=1000)#n_iter=100

        Y = ts.fit_transform(X_new)
        for cn, name in enumerate(names):
            idx = [n for n, d in enumerate(label)
                   if d == name]
            if name == 0:
                plt.scatter(Y[idx, 0], Y[idx, 1], c='r', marker='o', s=20, label='Healthy train')  # 2-dimensio
            elif name == 1:
                plt.scatter(Y[idx, 0], Y[idx, 1], c='g', marker='o', s=20, label='Schizophrenia train')  # 2-dimensio
            elif name == 2:
                plt.scatter(Y[idx, 0], Y[idx, 1], c='y', marker='s', s=20,
                            label='Healthy test')  # 2-dimensio   edgecolors='r',
            else:
                plt.scatter(Y[idx, 0], Y[idx, 1], c='b', marker='D', s=20,
                            label='Schizophrenia test')  # 2-dimensio  edgecolors='g',
    plt.legend(bbox_to_anchor=(0.6, 1.), ncol=1, loc=2, shadow=True)
    plt.title('t-SNE visualization of the last hidden layer \n representation in the DNN for 2 classes')
    plt.show()
def fix_perplexity_tsne():
    plt.figure(figsize=(10, 7))#
    # plt.subplot(111)  #,axisbg='k'
    ts = TSNE(n_components=2, perplexity=50, init='pca', random_state=0, n_iter=1000)  # n_iter=100

    Y = ts.fit_transform(X_new)
    for cn, name in enumerate(names):
        idx = [n for n, d in enumerate(label)
               if d == name]
        if name==0:
            plt.scatter(Y[idx, 0], Y[idx, 1], c = 'r',marker='o',  s=20, label='cMCI')  # 2-dimensio
        if name == 1:
            plt.scatter(Y[idx, 0], Y[idx, 1], c = 'g',marker='o',  s=20, label='AD')  # 2-dimensio
        if name == 2:
            plt.scatter(Y[idx, 0], Y[idx, 1], c = 'y',marker='s',  s=20, label='HC')  # 2-dimensio   edgecolors='r',
        if name == 3:
            plt.scatter(Y[idx, 0], Y[idx, 1], c = 'b',marker='D',  s=20, label='MCI')  # 2-dimensio  edgecolors='g',
    plt.legend(bbox_to_anchor=(0.6, 1.), ncol=1, loc=2, shadow=True)
    plt.title('t-SNE visualization of the last hidden layer \n representation in the CNN for 2 classes')
    plt.show()
def rem_zscore_over3(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]>10:
                data[i,j] = np.mean(data[:,j])
    return data

data,label = load_data()
test = load_data(train_data=0)
data[152][68] = data[152][68]/1000
feature_selection = range(data.shape[1])

index2remove = np.array([38,40,41,43,44,71])-3
index2mul1000 =np.array(range(69,139) + range(277,413))
index2div1000 = np.array(range(413,429))

for i in index2mul1000:
    for j in range(data.shape[0]):
        if data[j,i] < 10:
            data[j,i] = data[j,i] * 1000.0
for i in index2div1000:
    for j in range(data.shape[0]):
        if data[j, i] > 10000:
            data[j, i] = data[j, i] / 1000.0
for i in range(len(index2remove)):
    feature_selection.remove(index2remove[i])
for i in index2mul1000:
    for j in range(test.shape[0]):
        if test[j,i] < 10:
            test[j,i] = test[j,i] * 1000.0
for i in index2div1000:
    for j in range(test.shape[0]):
        if test[j, i] > 10000:
            test[j, i] = test[j, i] / 1000.0
data = data[:,feature_selection]
test = test[:,feature_selection]
mean_train = np.mean(data,axis = 0)
mean_test = np.mean(test,axis=0)
test = test - (mean_test-mean_train)


data = z_score(data)
data = np.tanh(data)
label = label.ravel()
test = z_score(test)
test = np.tanh(test)

X = data
Y = np.zeros([240,4])
for i in range(240):
    Y[i, label[i]] = 1
testX = test

# Building the encoder
encoder = tflearn.input_data(shape=[None, 423])
encoder = tflearn.fully_connected(encoder, 50 )
encoder = tflearn.fully_connected(encoder, 30)

# Building the decoder
decoder = tflearn.fully_connected(encoder, 50)
decoder = tflearn.fully_connected(decoder, 423)

# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=None)
# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, X, n_epoch=100, validation_set=(data, data),
          run_id="auto_encoder", batch_size=20)

# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)
print(encoding_model.predict([X[0]]))

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")
# testX = tflearn.data_utils.shuffle(testX)[0]
# Applying encode and decode over test set
encode_decode = model.predict(testX)



result_str = []
for index,value in enumerate(result):
    if value==0:
        result_str.append('cMCI')
    if value==1:
        result_str.append('AD')
    if value==2:
        result_str.append('HC')
    if value==3:
        result_str.append('MCI')
print result_str
writer = pd.ExcelWriter('result.xlsx')
df = pd.DataFrame(result_str)
df.to_excel(writer,sheet_name='sample_submission')






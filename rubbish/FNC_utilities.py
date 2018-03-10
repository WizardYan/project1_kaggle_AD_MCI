# -*- coding: utf-8 -*-
import tensorflow as tf
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
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


from sklearn import svm,preprocessing
from sklearn import datasets,svm,feature_selection
from sklearn.metrics import *
from sklearn.datasets import make_classification,load_iris,make_multilabel_classification
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel,RFECV,VarianceThreshold,SelectKBest,SelectPercentile,chi2,f_classif,f_regression
from sklearn.model_selection import  StratifiedKFold,train_test_split,GridSearchCV,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, IncrementalPCA,SparsePCA,FastICA,MiniBatchSparsePCA
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_decomposition import CCA

def load_data(one_hot=True,regress_site=False,predict_other_site = False, site_index=0, half = True, data_path = 'data/result_postproc20160908.mat',label_path = 'data/973_multi_site_subjects_1100_20160301.xlsx'):
    '''
    This function is used to laod the FNC data

    :return: data_sampleXfeature, Y
    '''

    origin_data = scio.loadmat(data_path)  # the FNC of 973 fMRI data, for more information,consult Shengfeng.liu

    if data_path=='data/result_postproc20160908.mat':
        data = origin_data['C_subj'][0] # data format : samples * length * width
    if data_path == 'data/result_uni_anal_20160921.mat':
        data = origin_data['c_subj_regressed'][0]
    if data_path=='data/cobre.mat':
        data = origin_data['cobre']
    if data_path=='data/Brain_atlas_fnc.mat':
        data = origin_data['C_subj'][0]

    if half == False:
        data_sampleXfeature = np.zeros((len(data),data[0].size))
    if half == True:
        data_sampleXfeature = np.zeros((len(data),len(data[0])*(len(data[0])-1)/2))

    origin_excel = pd.read_excel(label_path,'Sheet1',header=0) # this excel contains the label
    label = np.zeros((len(origin_excel['Group']),1),dtype='int32')  # word label

    for i in range(len(origin_excel['Group'])):

        if half ==True:  # half the data
            data_temp = []
            for m in range(1,len(data[0])):
                for n in range(m):
                    data_temp.append(data[i][m][n])
            data_sampleXfeature[i] = data_temp
        if half == False:
            data_sampleXfeature[i] = np.reshape(data[i],data[0].size) #samples * length * width ->> samples * features

        if origin_excel['Group'][i] == 'NC':   # change the world label to None_one_hot label
            label[i] = 0
        else:
            label[i] = 1
    label = label.ravel()

    site = []
    for j in set(origin_excel['Site']):  # site number
        site_temp = []
        for i in range(len(origin_excel['Site'])):
            if origin_excel['Site'][i] == j:
                site_temp.append(i)
        site.append(site_temp)

    #==== regress site effect
    if regress_site:
        mean = np.zeros([len(site), data[0].size])
        for i in range(len(site)):
            mean[i] = np.mean(data_sampleXfeature[site[i]], axis=0)
        mean_all = np.mean(mean, axis=0)
        bias_mean = mean - mean_all

        for j in range(len(site)):
            data_sampleXfeature[site[i]] -= bias_mean[j]
    if predict_other_site:
        site_one = data_sampleXfeature[site[site_index]]  # the site to predict
        label_site_one = label[site[site_index]]
        site_others = np.delete(data_sampleXfeature, site[site_index], axis=0) # the site to train
        label_site_others = np.delete(label, site[site_index])

        index = range(len(label_site_others))

        # np.random.seed(1)
        np.random.shuffle(index)
        site_others = site_others[index]
        label_site_others = label_site_others[index]
        nb_classes = 2

        Y_others = np.zeros((len(label_site_others),nb_classes), dtype='int32')
        for j in range(len(label_site_others)):
            Y_others[j, label_site_others[j]] = 1
        Y_one = np.zeros((len(label_site_one),nb_classes), dtype='int32')
        for j in range(len(label_site_one)):
            Y_one[j, label_site_one[j]] = 1

    # shuffle the data
    index = range(len(label))
    # np.random.seed(121)
    np.random.shuffle(index)
    data_sampleXfeature = data_sampleXfeature[index]
    label = label[index]

    # Change the None_one_hot label to one-hot label
    nb_classes = 2
    Y = np.zeros((len(label), nb_classes),dtype='int32')

    for j in range(len(label)):
        Y[j, label[j]] = 1

    if not predict_other_site:
        if one_hot == False:
            return data_sampleXfeature,label
        else:
            return data_sampleXfeature,Y

    if predict_other_site:
        if one_hot == False:
            return site_others,label_site_others,site_one,label_site_one
        else:
            return site_others,Y_others,site_one,Y_one

def Next(data,num_batch,batch_size):
    '''
    we want to create a index table
    :param data: samples * features
    :param num_batch: how much batch we want to input
    :param batch_size: the size of each batch
    :return: index
    '''

    len_data = len(data)
    index_table = np.zeros((num_batch,batch_size),dtype='int64')
    batch_size = batch_size;
    for i in range(num_batch):
        index = range((i*batch_size)%len_data,(i+1)*batch_size%len_data)
        if (i+1)*batch_size%len_data < i*batch_size%len_data:
          over = (i+1)*batch_size%len_data
          index = range(i*batch_size%len_data,len_data) + range(0,over)
        index_table[i] = np.transpose(index)
    return index_table

def label2onehot(label,nb_classes):
    onehot_label = np.zeros((len(label),nb_classes),dtype='int32')
    for i in range(len(label)):
        onehot_label[label[i]] = 1
    return onehot_label

def train_test_split_yan(data,label,batch_index,cross_num=10,validation=False):
    '''

    :param data:
    :param label:
    :param batch_index: the index of the test data
    :param cross_num: default is 10, you can choose from 1-10
    :param validation:
     :return:

    example:

        data = np.random.randint(1,10,size=[30,10])
        label = np.ones((30,1))
        batch_index = 3
        X_train,Y_train,X_test,Y_test = train_test_split_yan(data,label,batch_index,cross_num=5)
        print X_train
    '''
    if batch_index > cross_num:
        raise ValueError('batch index should never be bigger than cross_num')
    # This function is used for cross-validation
    # The index should be from 0 to 9
    num_subjects = data.shape[0]
    batch_size = num_subjects/cross_num

    X_train = np.row_stack((data[0:batch_size*batch_index],data[batch_size*(batch_index+1):num_subjects]))
    y_train = np.concatenate((label[0:batch_size*batch_index],label[batch_size*(batch_index+1):num_subjects]))

    X_test = data[batch_size*batch_index:batch_size*(batch_index+1)]
    y_test = label[batch_size*batch_index:batch_size*(batch_index+1)]


    if validation:
        num_subjects = data.shape[0]
        batch_size = num_subjects / (cross_num+1)

        X_train = np.row_stack((data[0:batch_size * batch_index], data[batch_size * (batch_index + 2):num_subjects]))
        y_train = np.concatenate((label[0:batch_size * batch_index], label[batch_size * (batch_index + 2):num_subjects]))

        X_test = data[batch_size * batch_index:batch_size * (batch_index + 1)]
        y_test = label[batch_size * batch_index:batch_size * (batch_index + 1)]

        X_validation = data[batch_size * (batch_index+1):batch_size * (batch_index + 2)]
        y_validation = label[batch_size * (batch_index+1):batch_size * (batch_index + 2)]
        return X_train,y_train,X_test,y_test,X_validation,y_validation
    else:
        return X_train,y_train,X_test,y_test

def acc_sens_spec_f_numpy(y_pre,y_true):
    '''

    :param y_pre:
    :param y_true:
    :return:
    '''
    # accuracy
    correct_prediction = np.equal(y_pre, y_true)
    # float_correct_prediction = np.cast(correct_prediction, np.float32)
    accuracy = np.mean(correct_prediction)

    # sensitivity
    sens_total = np.sum(y_true)
    sens_temp = np.sum(y_pre*y_true)
    sensitivity = sens_temp/float(sens_total)
    # specificity
    spec_total = np.size(y_true)-sens_total
    spec_temp = np.sum( (np.ones(len(y_true))-y_pre) * (np.ones(len(y_true))-y_true) )
    specificity = spec_temp/float(spec_total)
    # F-score
    f_score = 2*(accuracy*sensitivity)/(accuracy+sensitivity)
    return accuracy,sensitivity,specificity,f_score

def LRP(Xj,W):
    '''
    :param Xj:      temp0 = np.load('data/mnist_hidden_layers_5.npy')
                    temp1 = np.load('data/mnist_hidden_layers_4.npy')
                    temp2 = np.load('data/mnist_hidden_layers_2.npy')
                    temp3 = np.load('data/mnist_hidden_layers_0.npy')
                    temp4 = data
                    Xj = [temp0[1],temp1[1],temp2[1],temp3[1],temp4[1]]

    :param W:       temp0 = clf.model.get_weights()[6]
                    temp1 = clf.model.get_weights()[4]
                    temp2 = clf.model.get_weights()[2]
                    temp3 = clf.model.get_weights()[0]
                    print temp3.shape
                    W =[temp0,temp1,temp2,temp3]
    :return:        R_origin, R_new
    '''

    for n in range(0, 4):
        xj = Xj[n]
        #   xj = np.load('data/mnist_hidden_layers_5.npy')
        #        xj = xj[1,] # this is the last layer
        # print 'The shape of the xj is : ', xj.shape

        xi = Xj[n + 1]
        #    xi = np.load('data/mnist_hidden_layers_4.npy')
        #        xi = xi[1,]
        # print 'The shape of the xi is : ', xi.shape

        w = W[n]
        #    w = clf.model.get_weights()[6]
        # print 'The weight matrix is:', w.shape

        i_num = w.shape[0]
        j_num = w.shape[1]

        z = np.zeros([i_num, j_num])
        for i in range(0, i_num):
            for j in range(0, j_num):
                z[i][j] = xi[i] * w[i][j]
        z_sum = np.sum(z, axis=0)

        fracz = np.zeros([i_num, j_num])
        for i in range(0, i_num):
            for j in range(0, j_num):
                z[i][j] = xi[i] * w[i][j]
                fracz[i][j] = z[i][j] / z_sum[j]

        back_ij = np.zeros([i_num, j_num])
        for j in range(0, j_num):
            for i in range(0, i_num):
                back_ij[i][j] = fracz[i][j] * xj[j]

        R_new = np.sum(back_ij, axis=1)
     #   print Xj[n + 1]
        R_origin = Xj[n + 1]
        Xj[n + 1] = R_new
       # print Xj[n + 1]
        #print '##################'
        #print Xj[n + 1].shape

    return R_origin, R_new


def load_mci_data(train_data = True,one_hot=True):
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
        label2num = label2num.ravel()
    data = origin_excel.values
    if train_data == 1:
        data = data[:,3:]  # keep data from "age" to "end"
    if train_data == 0:
        data = data[:,2:]
    data = np.hstack((gender2num,data))
    data_final = data.astype('float32')

    if train_data == 1 and one_hot == False:
        return data_final,label2num
    if train_data == 1 and one_hot == True:
        Y = np.zeros([240, 4])
        for i in range(240):
            Y[i, label2num[i]] = 1
        return data_final,Y
    if train_data == 0:
        return data_final



def rfe_sigma(data,sigma = 2):
    rfe = True
    i = 1
    j = 1
    mean = np.mean(data)
    data -= mean
    while rfe:
        mean = np.mean(data)
        std = np.std(data)

        index = np.argsort(data)
        if (i+j)== len(data):
            print 'so bad'


        #############################################
        # over = data[index[-i]] > (mean + sigma * std)
        # if over:
        # # if data[index[-i]] > (mean +  np.abs(mean)):
        #     data[index[-i:]] = data[index[-(i + 1)]]
        #     i = i + 1
        # down = data[index[j-1]] < (mean - sigma*std)
        # if down:
        # # if data[index[-i]] < (mean )
        #     data[index[0:j]] = data[index[j]]
        #     j = j + 1
        #
        # if (not over) and  (not down):
        #     rfe = False
        ##############################################
        over = data[index[-i]] > (mean + sigma * std)
        if over:
            data[index[-i]] = mean
            i = i + 1
        down = data[index[j - 1]] < (mean - sigma * std)
        if down:
            data[index[j]] = mean
            j = j + 1
        if (not over) and (not down):
            rfe = False
    print 'i is:',i, 'j is:', j

    return data
def median(data,up=0.75,down=0.25):
    data_median = np.median(data)
    data_up = np.percentile(data,up)
    data_down = np.percentile(data,down)
    print 'good'
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
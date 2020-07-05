# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 2017

@author: zwang@nankai.edu.cn

Description: test BP neural network
"""
import BPNnetwork
import pandas as pd
import numpy as np


# 1、将标签转化为BP神经网络输出神经元格式，例如0转化为001,1转化为010,2转化为100
def vectorized_result(j):

    e = np.zeros((3, 1))

    e[j] = 1.0

    return e


# ##########2、标签错位年度预测##################
def CuoWeiPrediction():
    
    data = pd.read_csv(filename, index_col=0)

    # 移位，数据取2004年-2018年
    feats = data.iloc[0:15, :]
    feats = feats.values

    predict = data.iloc[15]
    predict = predict.values
    # predict = np.array(np.reshape(predict, (4, 1)))       # 定量分析
    # predict = np.array(np.reshape(predict, (15, 1)))      # 定量分析，无pca

    predict = np.array(np.reshape(predict, (5, 1)))     # 定性分析

    label = [0, 0, 2, 1, 2, 2, 1, 1, 0, 0, 1, 1, 1, 1, 1, 2]
    # 标签取2005年-2019年
    label = label[1:]
    
    # 产生训练集和测试集
    x_train = feats
    y_train = label
    x_test = feats
    y_test = label

    # training_inputs = [np.reshape(x, (4, 1)) for x in X_train]    # 定量分析
    # training_inputs = [np.reshape(x, (15, 1)) for x in X_train]   # 定量分析，无pca

    training_inputs = [np.reshape(x, (5, 1)) for x in x_train]   # 定性分析

    training_results = [vectorized_result(y) for y in y_train]
    training_data = list(zip(training_inputs, training_results))
    # test_inputs = [np.reshape(x, (4, 1)) for x in X_test]    # 定量分析
    # test_inputs = [np.reshape(x, (15, 1)) for x in X_test]   # 定量分析，无pca

    test_inputs = [np.reshape(x, (5, 1)) for x in x_test]   # 定性分析
    test_data = list(zip(test_inputs, y_test))

    # 投入BP神经网络学习
    # net = BPNnetwork.Network([4, 70, 70, 70, 3])    # 定量分析
    # net = BPNnetwork.Network([15, 70, 70, 70, 3])   # 定量分析，无pca

    net = BPNnetwork.Network([5, 40, 40, 40, 3])    # 定性分析

    net.SGD(training_data, 50, 5, 1.0, test_data=test_data, predict_data=predict)


# ##########3、标签错位季度预测##################
def CuoWeiPrediction_sea(sea_n):
    """sea_n: 第n个季度"""
    data = pd.read_csv(filename, index_col=0)

    # 移位，数据取2004年-2018年, 标签取2005年-2019年
    sealist_data = [4*i+sea_n-1 for i in range(15)]
    sealist_tag = [4*(i+1)+sea_n-1 for i in range(15)]

    feats = data.iloc[sealist_data, 0:4]
    feats = feats.values
    label = data.iloc[sealist_tag, 4].values

    predict = data.iloc[60+sea_n-1, 0:4]
    predict = predict.values
    predict = np.array(np.reshape(predict, (4, 1)))     # 定量分析

    print(predict)

    # 产生训练集和测试集
    x_train = feats
    y_train = label
    x_test = feats
    y_test = label

    training_inputs = [np.reshape(x, (4, 1)) for x in x_train]      # 定量分析
    training_results = [vectorized_result(y) for y in y_train]
    training_data = list(zip(training_inputs, training_results))
    test_inputs = [np.reshape(x, (4, 1)) for x in x_test]           # 定量分析

    test_data = list(zip(test_inputs, y_test))

    # 投入BP神经网络学习
    net = BPNnetwork.Network([4, 60, 60, 60, 60, 60, 3])            # 定量分析
    net.SGD(training_data, 50, 5, 0.7, test_data=test_data, predict_data=predict)


if __name__ == "__main__":
    
    # filename = r'year_pca.csv'           # 定量分析
    # filename = r'year_nor.csv'           # 定量分析，无pca
    filename = r'tags_year_pca.csv'      # 定性分析
    CuoWeiPrediction()
# 
#     filename = r'season_pca.csv'   # 季度定量分析
#     sea_n = 4
#     CuoWeiPrediction_sea(sea_n)
#     print("季度: ", sea_n)

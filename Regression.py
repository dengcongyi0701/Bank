# -*- coding: utf-8 -*-
"""
Created on Nov 27 2017

@author: tmq5971@163.com

Dsription：the Regressor algorithms in sklearn.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, cross_validation
import scipy

plt.rcParams['font.sans-serif'] = ['SimHei']        # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False          # 解决保存图像是负号'-'显示为方块的问题


# ##########1.数据生成部分##########
def transformData(n):
    
    s = str(n)
    data = pd.read_csv(datafile)
    data = data[s]          # 第一个指标的数据
    data = data.values
    
    l = len(data)

    # for i in range(l):
    #     data[i] = math.log(data[i])
     
    # 初始化data samples的个数
    # 一个data sample有四个数据，前三季度预测第四季度
    array = [0.0, 0.0, 0.0, 0.0]
    empty = np.array([array] * l)

    for i in range(l-3):
        empty[i][0] = data[i]
        empty[i][1] = data[i+1]
        empty[i][2] = data[i+2]
        empty[i][3] = data[i+3]
    empty = pd.DataFrame(empty)

    x = empty.iloc[:, 0:3]
    y = empty.iloc[:, 3:4]

    # 用每年的四季度数据作为测试集，其他的作为训练集
    list_test = [4*i for i in range(16)]
    list_train = list()
    for i in range(60):
        if i % 4 != 0:
            list_train.append(i)

    x_test = x.iloc[list_test, [0, 1, 2]]
    y_test = y.iloc[list_test, [0]]
    x_train = x.iloc[list_train, [0, 1, 2]]
    y_train = y.iloc[list_train, [0]]

    return x_train, x_test, y_train, y_test


# ##########2.回归部分##########
def try_different_method(name, model):
    
    # x_train, x_test, y_train, y_test = transformData(n)
    #
    # model.fit(x_train, y_train)

    # 1、此回归模型的在每一个指标上的预测准确率
    # score = model.score(x_test, y_test)
    # print("current:", score)
    #
    # #2、回归模型的预测结果
    # result = model.predict(x_test)
    # plt.figure()
    #
    # plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    #
    # plt.xticks(range(0, 16), ('2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
    #            '2013', '2014', '2015', '2016', '2017', '2018', '2019'), rotation=40)
    #
    # plt.title('model:RandomForestRegressor, score: %f'%score)
    #
    # plt.legend()
    #
    # filename = ".\picture\\"+name+"("+str(n)+").png"
    #
    # plt.savefig(filename)
    #
    # plt.show()

    # 3、此回归模型在15个指标上的平均预测准确率
    avg = 0
    # f = open(".\picture\\" + name + ".txt", 'w')
    for n in range(1, 16):
        
        x_train, x_test, y_train ,y_test = transformData(n)
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)

        # result = model.predict(x_test)
        # plt.figure()
        # plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
        # plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
        # plt.xticks(range(0, 16), ('2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
        #                           '2013', '2014', '2015', '2016', '2017', '2018', '2019'), rotation=40)
        # plt.title('model:'+name+'('+str(n)+'), score: %f' % score)
        # plt.legend()
        # filename = ".\picture\\" + name + "(" + str(n) + ").png"
        #
        # plt.savefig(filename)

        avg += score
        print(score)
        # f.write(str(score)+'\n')
    avg = avg/15
    print("Avg:", avg)
    # f.write("Avg:"+str(avg))
    # f.close()


# ##########3.具体方法选择##########
# ###3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor(criterion="mse", min_weight_fraction_leaf=0.1)

# ###3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()

# ###3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR(kernel='linear', C=1.90)

# ###3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_neighbors=6, weights='distance')

# ###3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20, max_depth=15)         #这里使用20个决策树

# ###3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=100)                              #这里使用100个决策树

# ###3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.2,
                                                                     max_depth=1)                   #这里使用100个决策树

# ###3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor(n_estimators=100)

####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()


# ##########4.预测2020年的第一季度数据##########
def Mytest2020_1(model):
    result = list()
    for n in range(1, 16):
        s = str(n)
        data = pd.read_csv(datafile)
        data = data[s]
        x = data.iloc[61: 64]
        x = [x]

        x_train, x_test, y_train, y_test = transformData(n)
        model.fit(x_train, y_train)
        result.append(model.predict(x)[0])
    return result


# ##########5.预测2020年的第二季度数据##########
def Mytest2020_2(model, season1):
    result = list()
    for n in range(1, 16):
        s = str(n)
        data = pd.read_csv(datafile)
        data = data[s]
        x = data.iloc[62: 64]
        x = [np.append(x.values, season1[n-1])]

        x_train, x_test, y_train, y_test = transformData(n)
        model.fit(x_train, y_train)
        result.append(model.predict(x)[0])
    return result


# ##########6.根据2004-2019各季度样本求各参数的均值置信区间##########
def mean_interval():
    df = pd.read_csv(datafile)
    lower_limit = []
    upper_limit = []
    for n in range(1, 16):
        x = df[str(n)]
        std = np.std(x, ddof=1)     # 样本标准差
        mean = np.mean(x)           # 样本均值
        sum = len(x)                # 样本量
        confidence = 0.95           # 置信水平

        alpha = 1 - confidence
        z_score = scipy.stats.norm.isf(alpha / 2)           # z分布临界值
        t_score = scipy.stats.t.isf(alpha / 2, df=(n - 1))  # t分布临界值

        me = z_score * std / np.sqrt(n)
        a = round(mean-me, 3)
        b = round(mean+me, 3)
        lower_limit.append(a)
        upper_limit.append(b)

    return lower_limit, upper_limit

# ##########7.画出某个特定参数的样本曲线、预测曲线及置信区间##########
def draw(n):
    df = pd.read_csv(datafile)
    tr = df[str(n)]
    tr.index = [pd.date_range('2004-3', '2020-1', freq='3M')]
    season1 = Mytest2020_1(model_SVR)               # 预测2020一季度数据
    season2 = Mytest2020_2(model_SVR, season1)      # 预测2020二季度数据
    pre = pd.DataFrame([season1[n-1], season2[n-1]], index=[pd.date_range('2020-3', '2020-9', freq='3M')])
    lower_limit, upper_limit = mean_interval()      # 置信区间

    plt.plot(tr, linestyle="-", color="gray", label='样本值')
    plt.plot(pre, linestyle="-", color="red", label='预测值')

    plt.xticks(pd.date_range(tr.index[0], '2020-12', freq='12M'), rotation=40)
    plt.axhline(lower_limit[n - 1], ls=":", c="teal", label="置信区间上界")
    plt.axhline(upper_limit[n - 1], ls=":", c="teal", label="置信区间下界")

    title = r"2004-2020 实际美元指数折线图"
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    datafile = r'season.csv'

    # try_different_method("SVR", model_SVR)

    # season1 = Mytest2020_1(model_SVR)
    # season2 = Mytest2020_2(model_SVR, season1)
    n = 1               # 预测第几个指标的数据，n从1取到15.

    draw(n)


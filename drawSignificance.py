# -*- coding: utf-8 -*-
"""
Created on Nov 17 2017

@author: zwang@nankai.edu.cn

Dsription：draw data with significant level
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from collections import Counter
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# #####计算每个点的pdist数值和显著度#############
def FirstThree(l):
    l.sort()
    d1 = l[0]
    d2 = l[1]
    d3 = l[2]
    return d1, d2, d3


def pdist():
    
    data = pd.read_csv(datafile)
    data1 = data
    
    # 与高风险点越接近，越危险，数值越大
    # 提取高风险数据，作为一个集合
    level2 = data[data['L'] == 2]
    level2 = level2.drop(['L'], axis=1)
    l2 = level2.values

    data = data.drop(['L'], axis=1)
    data = data.values
     
    # 计算每一个样本与高风险集合的距离，取前三个最大距离的平均值作为pdist
    dis = []

    for x in data:
        temp_dis = []
        for y in l2:
            k = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])
            d1 = sqrt(k)
            temp_dis.append(d1)
   
        d1, d2, d3 = FirstThree(temp_dis)
        avg = (d1+d2+d3)/3
        dis.append(avg)
    
    data1['dis'] = dis

    level2 = data1[data1['L'] == 2]
    level2_score = level2['dis']
    
    # 计算显著度
    pv = []
    lenth = len(level2)

    for i in dis:
        count = 0
        for j in level2_score:
            if j > i:
                count += 1
        count = float(count) / lenth
        pv.append(count)
    
    data1['pv'] = pv
    
    return data1


# ###########画带有显著度的pca降维图###############
def draw_pv(df):
    
    res = pd.DataFrame()
    res[0] = df['0']
    res[1] = df['1']
    res[2] = df['2']
    res[3] = (12-df['dis'])/11
    # -------------------------
    # 用显著度上色
    # -------------------------
    
    res = res.values
    nearest = lambda x, s: s[np.argmin(np.abs(s-x))]
    f = np.arange(0, 2.1, 0.1)
    res[:, 3] = [nearest(x, f) for x in res[:, 3]]
    # cm = plt.cm.get_cmap('coolwarm')
    # ax = plt.axes(projection='3d')
    # ax.scatter(res[:, 0], res[:, 1], res[:, 2], c=res[:, 3], marker='o', cmap=cm, s=80)

    cm = pl.cm.get_cmap('coolwarm')
    pl.axes(projection='3d')
    pl.scatter(res[:, 0], res[:, 1], res[:, 2], c=res[:, 3], marker='o', cmap=cm)
    pl.colorbar()

    plt.show()


if __name__ == "__main__":

    datafile = r'season_pca.csv'

    df = pdist()
    
    draw_pv(df)             # 画季度数据的显著性统计三维图

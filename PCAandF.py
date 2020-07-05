# -*- coding: utf-8 -*-

"""
Created on Nov 15 2017

@author: zwang@nankai.edu.cn

desription: calculate PCA and F score.
"""

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib.ticker as ticker


# ###############读取数据###############
def readData():
    data = pd.read_csv(datafile)
    data = data.values
    return data


# ###############归一化数据###############
def normalizeData(data):
    nor_data = normalize(data)
    nor_data = pd.DataFrame(nor_data)
    # nor_data = nor_data.round(4)
    # nor_data = nor_data.values
    # print (nor_data)
    nor_data.to_csv(nor_file)
    return nor_data


# ###############PCA降维,并计算F得分###############
def pcaAndF(nor_data):
    
    # pca= PCA(n_components=3)                #保留n个主成分
    pca = PCA(n_components=0.85)            # n_components取小数表示保证85%的信息保留率
    pca_data = pca.fit_transform(nor_data)

    ratio = pca.explained_variance_ratio_
    print(ratio)
    summ = sum(ratio)
    print("F1: {}, F2: {}, F3: {}".format(ratio[0], ratio[1], ratio[2]))
    # print("F1: {}, F2: {}, F3: {}, F4: {}".format(ratio[0], ratio[1], ratio[2], ratio[3]))
    print("SUM: {}".format(summ))
    
    F_score = []
    label = []
    n = len(ratio)

    for i in pca_data:
        
        s = 0
        for j in range(n):
            s = s+ratio[j]*i[j]

        s = round(s, 3)         # F得分保留三位小数
        F_score.append(s)
        if -0.030 <= s <= 0.030:
            label.append(0)
        elif -0.080 <= s < -0.030 or 0.030 < s <= 0.080:
            label.append(1)
        else:
            label.append(2)
        
    pca_data = pd.DataFrame(pca_data)
    pca_data['F'] = F_score
    # pca_data['L'] = label      # 定量分析中，用来标记
    draw(pca_data)
    pca_data.to_csv(pca_file)


# ####################画pca降维图(二维)################
def draw(pca):

    pca.index = [i for i in range(0, 64)]           # 季度数据
    # pca.index = [i for i in range(2004, 2020)]    # 年度数据

    d = pca[0]
    plt.plot(d, 'r.-', label='F1')

    d = pca[1]
    plt.plot(d, 'y*-', label='F2')

    d = pca[2]
    plt.plot(d, 'b1-', label='F3')

    plt.legend()
    plt.show() 
    
if __name__ == "__main__":
    
    datafile = r'year.csv'           # 原数据路径
    nor_file = r'year_nor.csv'       # 归一化后的数据路径
    pca_file = r'year_pca.csv'       # pca后的数据路径
    # pca_file = r'year_pca_f4.csv'

    # datafile = r'tags_year.csv'       # 定性分析原数据
    # pca_file = r'tags_year_pca.csv'   # 定性分析pca后数据
    # data = readData()                 # 定性分析无需归一化
    # pcaAndF(data)

    # datafile = r'season.csv'          # 季度数据
    # nor_file = r'season_nor.csv'      # 归一化季度数据
    # pca_file = r'season_pca.csv'      # pca季度数据

    data = readData()
    nor_data = normalizeData(data)
    pcaAndF(nor_data)

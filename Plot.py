# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 15:05:44 2017

@author: zwang@nankai.edu.cn

Description: draw data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# ########1.折线图(绘制年度F值)###############
def zhexian():

    df = pd.read_csv(filename)
    df = df['F']
    df.index = [i for i in range(2004, 2020)]
    print(df)
    plt.axhline(0.025, ls=":", c="orange", label="预警阈值(±0.025)")
    plt.axhline(-0.025, ls=":", c="orange")
    plt.axhline(0.070, ls=":", c="red", label="危险阈值(±0.070)")
    plt.axhline(-0.070, ls=":", c="red")
    plt.plot(df, linestyle="-", color="gray", label='F')
    for year in df.index:
        if -0.025 <= df[year] <= 0.025:
            plt.plot(year, df[year], color="teal", marker="*")
        elif -0.070 <= df[year] < -0.025 or 0.025 < df[year] <= 0.070:
            plt.plot(year, df[year], color="orange", marker="*")
        else:
            plt.plot(year, df[year], color='red',marker="*")

    plt.text(2004, -0.035, "安全", fontsize=10, color="teal", style="italic", horizontalalignment='center')
    plt.text(2005.2, 0.005, "安全", fontsize=10, color="teal", style="italic", horizontalalignment='left')
    plt.text(2006, 0.095, "危险", fontsize=10, color="red", style="italic", horizontalalignment='center')
    plt.text(2007, 0.036, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2008.2, 0.065, "危险", fontsize=10, color="red", style="italic", horizontalalignment='left')
    plt.text(2009.2, 0.216, "危险", fontsize=10, color="red", style="italic", horizontalalignment='left')
    plt.text(2010, -0.069, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2011.1, -0.046, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='left')
    plt.text(2012, -0.009, "安全", fontsize=10, color="teal", style="italic", horizontalalignment='center')
    plt.text(2013, -0.019, "安全", fontsize=10, color="teal", style="italic", horizontalalignment='center')
    plt.text(2014, -0.022, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2015, -0.021, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2016, -0.069, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2017, -0.078, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2018, -0.040, "预警", fontsize=10, color="orange", style="italic", horizontalalignment='center')
    plt.text(2019.2, -0.069, "危险", fontsize=10, color="red", style="italic", horizontalalignment='center')

    plt.legend()
    plt.show()


# ########1.5 折线图(绘制季度F值)###############
def zhexian_season():

    df = pd.read_csv(filename)
    df = df['F']
    df.index = [i for i in range(0, 64)]
    print(df)
    plt.axhline(0.030, ls=":", c="orange")
    plt.axhline(-0.030, ls=":", c="orange")
    plt.axhline(0.080, ls=":", c="red")
    plt.axhline(-0.080, ls=":", c="red")
    plt.plot(df, linestyle="-", color="gray", label='F')
    for year in df.index:
        if -0.025 <= df[year] <= 0.025:
            plt.plot(year, df[year], color="teal", marker="o", label="安全")
        elif -0.070 <= df[year] < -0.025 or 0.025 < df[year] <= 0.070:
            plt.plot(year, df[year], color="orange", marker="o", label="预警")
        else:
            plt.plot(year, df[year], color='red', marker="o", label="危险")

    # 不显示重复的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()


# ###############2.热点图###############
def redian():

    df = pd.read_csv(filename, index_col=0)
    datav = df.values
    dataz = scale(datav)
    sns.heatmap(dataz)
    plt.legend()
    plt.show()


# ###############3.箱线散点图###############
def xiangxian():

    df = pd.read_csv(filename, index_col=0)
    print (df)
    # ax = sns.boxplot(x="label", y="D1", data=df)
    ax = sns.boxplot(data=df)
    # ax = sns.swarmplot(x="label", y="D1", data=df, color=".25")
    ax = sns.swarmplot(data=df, color=".25")
    plt.legend()
    plt.show()


# ###############4、三维图###############
def sanwei():

    df = pd.read_csv(filename, index_col=0)
    ax = plt.subplot(111, projection='3d')      # 创建一个三维的绘图工程

    # 将数据点分成三部分画，在颜色上有区分度
    lev1 = {}
    lev2 = {}
    lev3 = {}
    for i in range(3):
        lev1[i] = list()
        lev2[i] = list()
        lev3[i] = list()
    for year in df.index:
        if -0.030 <= df['F'][year] <= 0.030:
            lev1[0].append(df['0'][year])
            lev1[1].append(df['1'][year])
            lev1[2].append(df['2'][year])
        elif -0.080 <= df['F'][year] < -0.030 or 0.030 < df['F'][year] <= 0.080:
            lev2[0].append(df['0'][year])
            lev2[1].append(df['1'][year])
            lev2[2].append(df['2'][year])
        else:
            lev3[0].append(df['0'][year])
            lev3[1].append(df['1'][year])
            lev3[2].append(df['2'][year])
    print(lev3[0], lev2[0], lev1[0])
    ax.scatter(lev3[0], lev3[1], lev3[2], c='r', label='level 3', s=50)
    ax.scatter(lev2[0], lev2[1], lev2[2], c='orange', label='level 2', s=50)
    ax.scatter(lev1[0], lev1[1], lev1[2], c='teal', label='level 1', s=50)

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.legend()
    plt.show()    

    
if __name__ == "__main__":
    
    filename = r'./year_pca.csv'  #年度数据
    # filename = r'./season_pca.csv'  #季度数据

    zhexian()
    # zhexian_season()
    # redian()
    # xiangxian()
    # sanwei()

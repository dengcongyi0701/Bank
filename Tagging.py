"""
用15个指标自动打标签
"""
import pandas as pd


def tag(file, tagfile_name):
    """
    根据1.doc文档中总结的各个标签的数值进行划分
    :param file:
    :return:
    """
    data = pd.read_csv(file)
    data = data.values
    Tag = dict()
    for i in range(0, 16):   # 2004年到2019年
        j = i+2004
        Tag[j] = list()

        # GDP,0安全,1低风险,2中风险,3高风险
        if 6.9 <= data[i][0] < 9.5:
            Tag[j].append(0)
        elif 5 <= data[i][0] < 6.5 or 9.5 <= data[i][0] < 11.5:
            Tag[j].append(1)
        elif 3.5 <= data[i][0] < 5 or 11.5 <= data[i][0] < 12.5:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # CPI
        if data[i][1] < 105:
            Tag[j].append(0)
        elif 105 <= data[i][1] < 110:
            Tag[j].append(1)
        elif 110 <= data[i][1] < 115:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # M2增长率
        if 5 <= data[i][2] < 15:
            Tag[j].append(0)
        elif 15 <= data[i][2] < 20:
            Tag[j].append(1)
        elif 20 <= data[i][2] < 25:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 固定资产投资增长率
        if 13 <= data[i][3] or data[i][3] < 19:
            Tag[j].append(0)
        elif 10 <= data[i][3] < 13 or 19 <= data[i][3] < 22:
            Tag[j].append(1)
        elif 7 <= data[i][3] < 10 or 22 <= data[i][3] < 25:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 股票市盈率
        if data[i][4] < 40:
            Tag[j].append(0)
        elif 40 <= data[i][4] < 60:
            Tag[j].append(1)
        elif 60 <= data[i][4] < 80:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 房价增长率/GDP
        if 0 <= data[i][5] < 1:
            Tag[j].append(0)
        elif 1 <= data[i][5] < 2:
            Tag[j].append(1)
        elif 2 <= data[i][5] < 3:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 银行不良贷款率
        if data[i][6] < 12:
            Tag[j].append(0)
        elif 12 <= data[i][6] < 17:
            Tag[j].append(1)
        elif 17 <= data[i][6] < 22:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 各项贷款余额
        if data[i][7] < 15:
            Tag[j].append(0)
        elif 15 <= data[i][7] < 20:
            Tag[j].append(1)
        elif 20 <= data[i][7] < 25:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 经常项目差额/GDP
        if data[i][8] < 5:
            Tag[j].append(0)
        elif 5 <= data[i][8] < 7:
            Tag[j].append(1)
        elif 7 <= data[i][8] < 9:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 外汇储备/进口金额
        if 6 <= data[i][9]:
            Tag[j].append(0)
        elif 4 <= data[i][9] < 6:
            Tag[j].append(1)
        elif 3 <= data[i][9] < 4:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 短期债务/外债余额
        if data[i][10] < 25:
            Tag[j].append(0)
        elif 25 <= data[i][8] < 50:
            Tag[j].append(1)
        elif 50 <= data[i][8] < 70:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 外债总额/外汇储备
        if data[i][11] < 25:
            Tag[j].append(0)
        elif 25 <= data[i][11] < 35:
            Tag[j].append(1)
        elif 35 <= data[i][11] < 45:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 实际有效汇率指数
        if 95 <= data[i][12] < 105:
            Tag[j].append(0)
        elif 80 <= data[i][12] < 95 or 105 <= data[i][12] < 120:
            Tag[j].append(1)
        elif 70 <= data[i][12] < 80 or 120 <= data[i][12] < 135:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 国债收益率
        if 0 <= data[i][13] < 1:
            Tag[j].append(0)
        elif 1 <= data[i][13] < 2:
            Tag[j].append(1)
        elif 2 <= data[i][13] < 3:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

        # 实际美元指数
        if 90 <= data[i][14] < 95:
            Tag[j].append(0)
        elif 85 <= data[i][12] < 90 or 95 <= data[i][12] < 100:
            Tag[j].append(1)
        elif 80 <= data[i][12] < 85 or 100 <= data[i][12] < 105:
            Tag[j].append(2)
        else:
            Tag[j].append(3)

    df = pd.DataFrame(Tag)
    df = pd.DataFrame(df.values.T, columns=df.index)
    df.to_csv(tagfile_name, index=False)


if __name__ == '__main__':
    yearfile_name = r'year.csv'
    tagfile_name = r'tags_year.csv'
    tag(yearfile_name, tagfile_name)

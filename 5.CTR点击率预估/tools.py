import pandas as pd
import numpy as np
import scipy as sp

'''
    辅助函数库
'''


# 文件读取
def read_csv_file(f, logging=False):
    print('=======读取数据=======', f)
    data = pd.read_csv(f)
    if logging:
        print(data.head(5))
        print(f, "  包含以下列....")
        print(data.columns.values)
        print(data.describe())
        print(data.info())
    return data


# 第一类编码
def categories_process_first_class(cate):
    cate = str(cate)
    if len(cate) == 1:
        if int(cate) == 0:
            return 0
    else:
        return int(cate[0])


# 第2类编码
def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate) < 3:
        return 0
    else:
        return int(cate[1:])


# 年龄处理，切段
def age_process(age):
    age = int(age)
    if age == 0:
        return 0
    elif age < 15:
        return 1
    elif age < 25:
        return 2
    elif age < 40:
        return 3
    elif age < 60:
        return 4
    else:
        return 5


# 省份处理
def process_province(hometown):
    hometown = str(hometown)
    province = int(hometown[0:2])
    return province


# 城市处理
def process_city(hometown):
    hometown = str(hometown)
    if len(hometown) > 1:
        province = int(hometown[2:])
    else:
        province = 0
    return province


# 时间处理
def get_time_day(t):
    t = str(t)
    t = int(t[0:2])
    return t


# 一天切成4段
def get_time_hour(t):
    t = str(t)
    t = int(t[2:4])
    if t < 6:
        return 0
    elif t < 12:
        return 1
    elif t < 18:
        return 2
    else:
        return 3


# 评估与计算logloss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

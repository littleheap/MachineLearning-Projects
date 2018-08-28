from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tools import *

'''
    特征工程+随机森林建模
'''

# ['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
train_data = read_csv_file('./data/train.csv', logging=True)

'''
       label  clickTime  conversionTime  creativeID   userID  positionID  \
    0      0     170000             NaN        3089  2798058         293   
    1      0     170000             NaN        1259   463234        6161   
    2      0     170000             NaN        4465  1857485        7434   
    3      0     170000             NaN        1004  2038823         977   
    4      0     170000             NaN        1887  2015141        3688   
    
       connectionType  telecomsOperator  
    0               1                 1  
    1               1                 2  
    2               4                 1  
    3               1                 1  
    4               1                 1  
'''

# ['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']
ad = read_csv_file('./data/ad.csv', logging=True)

'''
       creativeID  adID  camgaignID  advertiserID  appID  appPlatform
    0        4079  2318         147            80     14            2
    1        4565  3593         632             3    465            1
    2        3170  1593         205            54    389            1
    3        6566  2390         205            54    389            1
    4        5187   411         564             3    465            1
'''

# app
app_categories = read_csv_file('./data/app_categories.csv', logging=True)

app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(categories_process_first_class)

app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(categories_process_second_class)

'''
       appID  appCategory
    0     14            2
    1     25          203
    2     68          104
    3     75          402
    4     83          203
'''

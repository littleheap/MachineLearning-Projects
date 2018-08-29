import os
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from tools import *
import warnings

os.environ["OMP_NUM_THREADS"] = "8"  # 并行训练
rng = np.random.RandomState(4315)
warnings.filterwarnings("ignore")

'''
    特征工程+XGBoost
'''

############################################################################
############################################################################

# train ['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
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

'''
                  label     clickTime  conversionTime    creativeID        userID  \
    count  3.749528e+06  3.749528e+06    93262.000000  3.749528e+06  3.749528e+06   
    mean   2.487300e-02  2.418317e+05   242645.358013  3.261575e+03  1.405349e+06   
    std    1.557380e-01  3.958793e+04    39285.385532  1.829643e+03  8.088094e+05   
    min    0.000000e+00  1.700000e+05   170005.000000  1.000000e+00  1.000000e+00   
    25%    0.000000e+00  2.116270e+05   211626.000000  1.540000e+03  7.058698e+05   
    50%    0.000000e+00  2.418390e+05   242106.000000  3.465000e+03  1.407062e+06   
    75%    0.000000e+00  2.722170e+05   272344.000000  4.565000e+03  2.105989e+06   
    max    1.000000e+00  3.023590e+05   302359.000000  6.582000e+03  2.805118e+06   

             positionID  connectionType  telecomsOperator  
    count  3.749528e+06    3.749528e+06      3.749528e+06  
    mean   3.702799e+03    1.222590e+00      1.605879e+00  
    std    1.923724e+03    5.744428e-01      8.491127e-01  
    min    1.000000e+00    0.000000e+00      0.000000e+00  
    25%    2.579000e+03    1.000000e+00      1.000000e+00  
    50%    3.322000e+03    1.000000e+00      1.000000e+00  
    75%    4.896000e+03    1.000000e+00      2.000000e+00  
    max    7.645000e+03    4.000000e+00      3.000000e+00  
'''

train_data['clickTime_day'] = train_data['clickTime'].apply(get_time_day)

train_data['clickTime_hour'] = train_data['clickTime'].apply(get_time_hour)

############################################################################
############################################################################

# ad ['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']
ad = read_csv_file('./data/ad.csv', logging=True)

'''
       creativeID  adID  camgaignID  advertiserID  appID  appPlatform
    0        4079  2318         147            80     14            2
    1        4565  3593         632             3    465            1
    2        3170  1593         205            54    389            1
    3        6566  2390         205            54    389            1
    4        5187   411         564             3    465            1
'''

'''
            creativeID         adID   camgaignID  advertiserID        appID  \
    count  6582.000000  6582.000000  6582.000000   6582.000000  6582.000000   
    mean   3291.500000  1786.341689   313.397144     44.381191   310.805682   
    std    1900.204068  1045.890729   210.636055     24.091342   125.577377   
    min       1.000000     1.000000     1.000000      1.000000    14.000000   
    25%    1646.250000   882.250000   131.000000     26.000000   205.000000   
    50%    3291.500000  1771.000000   274.000000     54.000000   389.000000   
    75%    4936.750000  2698.750000   512.000000     57.000000   421.000000   
    max    6582.000000  3616.000000   720.000000     91.000000   472.000000   

           appPlatform  
    count  6582.000000  
    mean      1.448952  
    std       0.497425  
    min       1.000000  
    25%       1.000000  
    50%       1.000000  
    75%       2.000000  
    max       2.000000  
'''

############################################################################
############################################################################

# app_categories ['appID' 'appCategory']
app_categories = read_csv_file('./data/app_categories.csv', logging=True)

'''
       appID  appCategory
    0     14            2
    1     25          203
    2     68          104
    3     75          402
    4     83          203
'''

'''
                   appID    appCategory 
    count  217041.000000  217041.000000
    mean   137220.306472     161.856133
    std    105340.872671     157.746571
    min        14.000000       0.000000
    25%     54585.000000       0.000000
    50%    111520.000000     106.000000
    75%    195882.000000     301.000000
    max    433269.000000     503.000000
'''

app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(categories_process_first_class)

app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(categories_process_second_class)

############################################################################
############################################################################

# User ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence']
user = read_csv_file('./data/user.csv', logging=True)

'''
       userID  age  gender  education  marriageStatus  haveBaby  hometown  \
    0       1   42       1          0               2         0       512   
    1       2   18       1          5               1         0      1403   
    2       3    0       2          4               0         0         0   
    3       4   21       2          5               3         0       607   
    4       5   22       2          0               0         0         0   

       residence  
    0        503  
    1       1403  
    2          0  
    3        607  
    4       1301  
'''

'''
                 userID           age        gender     education  marriageStatus  \
    count  2.805118e+06  2.805118e+06  2.805118e+06  2.805118e+06    2.805118e+06   
    mean   1.402560e+06  2.038662e+01  1.294072e+00  1.889235e+00    9.870540e-01   
    std    8.097680e+05  1.151120e+01  6.409864e-01  1.607085e+00    9.621890e-01   
    min    1.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00    0.000000e+00   
    25%    7.012802e+05  1.400000e+01  1.000000e+00  1.000000e+00    0.000000e+00   
    50%    1.402560e+06  2.000000e+01  1.000000e+00  2.000000e+00    1.000000e+00   
    75%    2.103839e+06  2.700000e+01  2.000000e+00  3.000000e+00    2.000000e+00   
    max    2.805118e+06  8.000000e+01  2.000000e+00  7.000000e+00    3.000000e+00   

               haveBaby      hometown     residence  
    count  2.805118e+06  2.805118e+06  2.805118e+06  
    mean   2.848418e-01  6.750372e+02  9.571084e+02  
    std    7.800834e-01  7.691699e+02  7.897154e+02  
    min    0.000000e+00  0.000000e+00  0.000000e+00  
    25%    0.000000e+00  0.000000e+00  3.020000e+02  
    50%    0.000000e+00  4.030000e+02  7.170000e+02  
    75%    0.000000e+00  1.201000e+03  1.507000e+03  
    max    6.000000e+00  3.401000e+03  3.401000e+03  
'''

# print(user.age.value_counts())

# print(user.columns)

user['age_process'] = user['age'].apply(age_process)

user["hometown_province"] = user['hometown'].apply(process_province)

user["hometown_city"] = user['hometown'].apply(process_city)

user["residence_province"] = user['residence'].apply(process_province)

user["residence_city"] = user['residence'].apply(process_city)

# print(user[user.age != 0].describe())

############################################################################
############################################################################

# test_data ['instanceID' 'label' 'clickTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
test_data = read_csv_file('./data/test.csv', True)

'''
       instanceID  label  clickTime  creativeID   userID  positionID  \
    0           1     -1     310000        3745  1164848        3451   
    1           2     -1     310000        2284  2127247        1613   
    2           3     -1     310000        1456  2769125        5510   
    3           4     -1     310000        4565     9762        4113   
    4           5     -1     310000          49  2513636        3615   

       connectionType  telecomsOperator  
    0               1                 3  
    1               1                 3  
    2               2                 1  
    3               2                 3  
    4               1                 3  
'''

'''
              instanceID     label      clickTime     creativeID        userID  \
    count  338489.000000  338489.0  338489.000000  338489.000000  3.384890e+05   
    mean   169245.000000      -1.0  311479.490613    3001.534765  1.409519e+06   
    std     97713.501971       0.0     580.393521    1869.336873  8.073083e+05   
    min         1.000000      -1.0  310000.000000       4.000000  3.000000e+00   
    25%     84623.000000      -1.0  311053.000000    1248.000000  7.149930e+05   
    50%    169245.000000      -1.0  311536.000000    3012.000000  1.411134e+06   
    75%    253867.000000      -1.0  311951.000000    4565.000000  2.108981e+06   
    max    338489.000000      -1.0  312359.000000    6580.000000  2.805117e+06   

              positionID  connectionType  telecomsOperator  
    count  338489.000000   338489.000000     338489.000000  
    mean     3640.126394        1.139015          1.629028  
    std      1902.559504        0.511882          0.854993  
    min         2.000000        0.000000          0.000000  
    25%      2436.000000        1.000000          1.000000  
    50%      3322.000000        1.000000          1.000000  
    75%      4881.000000        1.000000          2.000000  
    max      7645.000000        4.000000          3.000000  
'''

test_data['clickTime_day'] = test_data['clickTime'].apply(get_time_day)

test_data['clickTime_hour'] = test_data['clickTime'].apply(get_time_hour)

############################################################################
############################################################################

print(train_data.head())

'''
       label  clickTime  conversionTime  creativeID   userID  positionID  \
    0      0     170000             NaN        3089  2798058         293   
    1      0     170000             NaN        1259   463234        6161   
    2      0     170000             NaN        4465  1857485        7434   
    3      0     170000             NaN        1004  2038823         977   
    4      0     170000             NaN        1887  2015141        3688   

       connectionType  telecomsOperator  clickTime_day  clickTime_hour  
    0               1                 1             17               0  
    1               1                 2             17               0  
    2               4                 1             17               0  
    3               1                 1             17               0  
    4               1                 1             17               0  
'''

print(ad.head())

'''
       creativeID  adID  camgaignID  advertiserID  appID  appPlatform
    0        4079  2318         147            80     14            2
    1        4565  3593         632             3    465            1
    2        3170  1593         205            54    389            1
    3        6566  2390         205            54    389            1
    4        5187   411         564             3    465            1
'''

print(app_categories.head())

'''
       appID  appCategory  app_categories_first_class  app_categories_second_class
    0     14            2                         NaN                            0
    1     25          203                         2.0                            3
    2     68          104                         1.0                            4
    3     75          402                         4.0                            2
    4     83          203                         2.0                            3
'''

print(user.head())

'''
       userID  age  gender  education  marriageStatus  haveBaby  hometown  \
    0       1   42       1          0               2         0       512   
    1       2   18       1          5               1         0      1403   
    2       3    0       2          4               0         0         0   
    3       4   21       2          5               3         0       607   
    4       5   22       2          0               0         0         0   

       residence  age_process  hometown_province  hometown_city  \
    0        503            4                 51              2   
    1       1403            2                 14              3   
    2          0            0                  0              0   
    3        607            2                 60              7   
    4       1301            2                  0              0   

       residence_province  residence_city  
    0                  50               3  
    1                  14               3  
    2                   0               0  
    3                  60               7  
    4                  13               1 
'''

############################################################################
############################################################################

# 数据合并
train_user = pd.merge(train_data, user, on='userID')

train_user_ad = pd.merge(train_user, ad, on='creativeID')

train_user_ad_app = pd.merge(train_user_ad, app_categories, on='appID')

print(train_user_ad_app.head())

''' [5 rows x 30 columns]
   label  clickTime  conversionTime  creativeID   userID  positionID  \
0      0     170000             NaN        3089  2798058         293   
1      0     170001             NaN        3089   195578        3659   
2      0     170014             NaN        3089  1462213        3659   
3      0     170030             NaN        3089  1985880        5581   
4      0     170047             NaN        3089  2152167        5581   

   connectionType  telecomsOperator  clickTime_day  clickTime_hour  \
0               1                 1             17               0   
1               0                 2             17               0   
2               0                 3             17               0   
3               1                 1             17               0   
4               1                 1             17               0   

              ...               residence_province  residence_city  adID  \
0             ...                               13               1  1321   
1             ...                               13               1  1321   
2             ...                               13               1  1321   
3             ...                                0               0  1321   
4             ...                               13               3  1321   

   camgaignID  advertiserID  appID  appPlatform  appCategory  \
0          83            10    434            1          108   
1          83            10    434            1          108   
2          83            10    434            1          108   
3          83            10    434            1          108   
4          83            10    434            1          108   

   app_categories_first_class  app_categories_second_class  
0                         1.0                            8  
1                         1.0                            8  
2                         1.0                            8  
3                         1.0                            8  
4                         1.0                            8  

'''

print(train_user_ad_app.columns)

'''
    Index(['label', 'clickTime', 'conversionTime', 'creativeID', 'userID',
           'positionID', 'connectionType', 'telecomsOperator', 'clickTime_day',
           'clickTime_hour', 'age', 'gender', 'education', 'marriageStatus',
           'haveBaby', 'hometown', 'residence', 'age_process', 'hometown_province',
           'hometown_city', 'residence_province', 'residence_city', 'adID',
           'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory',
           'app_categories_first_class', 'app_categories_second_class'],
          dtype='object')
'''

############################################################################
############################################################################

# 取出数据和label

# 特征部分数据转换
x_user_ad_app = train_user_ad_app.loc[:,
                ['creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator', 'clickTime_day',
                 'clickTime_hour', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'residence',
                 'age_process', 'hometown_province', 'hometown_city', 'residence_province', 'residence_city', 'adID',
                 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'app_categories_first_class',
                 'app_categories_second_class']]

# value就是转化为numpy格式
x_user_ad_app = x_user_ad_app.values

x_user_ad_app = np.array(x_user_ad_app, dtype='int32')

# 标签部分数据转换
y_user_ad_app = train_user_ad_app.loc[:, ['label']].values

param_grid = {
    'max_depth': [3, 4, 5, 7, 9],
    'n_estimators': [10, 50, 100, 400, 800, 1000, 1200],
    'learning_rate': [0.1, 0.2, 0.3],
    'gamma': [0, 0.2],
    'subsample': [0.8, 1],
    'colsample_bylevel': [0.8, 1]
}

xgb_model = xgb.XGBClassifier()
rgs = GridSearchCV(xgb_model, param_grid, n_jobs=-1)
rgs.fit(x_user_ad_app, y_user_ad_app.reshape(y_user_ad_app.shape[0], ))
print(rgs.best_score_)
print(rgs.best_params_)

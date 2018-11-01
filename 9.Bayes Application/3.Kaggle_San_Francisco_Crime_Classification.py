import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# 用pandas载入csv训练数据，并解析第一列为日期格式
train = pd.read_csv('./train.csv', parse_dates=['Dates'])
test = pd.read_csv('./test.csv', parse_dates=['Dates'])

# 输出数据尺寸
print(train.shape)
# (878049, 9)

print(test.shape)
# (884262, 7)

# 数据预处理，形成one-hot形式数据
# 用LabelEncoder对不同的犯罪类型编号
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(train.Category)

# 因子化星期几，街区，小时等特征
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)

# 组合特征
trainData = pd.concat([hour, days, district], axis=1)
trainData['crime'] = crime

# 对于测试数据做相同的处理
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)

hour = test.Dates.dt.hour
hour = pd.get_dummies(hour)

testData = pd.concat([hour, days, district], axis=1)

print(trainData.shape)
# (878049, 42)

print(testData.shape)
# (884262, 41)

'''
    模型搭建
'''

# 只取星期几和街区作为分类器输入特征
features = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'BAYVIEW', 'CENTRAL',
            'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

# 分割训练集(3/5)和测试集(2/5)
training, validation = train_test_split(trainData, train_size=.60)

# 朴素贝叶斯建模，计算log_loss
model = BernoulliNB()
nbStart = time.time()
model.fit(training[features], training['crime'])
nbCostTime = time.time() - nbStart
predicted = np.array(model.predict_proba(validation[features]))
print("朴素贝叶斯建模耗时 %f 秒" % (nbCostTime))
# 朴素贝叶斯建模耗时 0.591072 秒
print("朴素贝叶斯log损失为 %f" % (log_loss(validation['crime'], predicted)))
# 朴素贝叶斯log损失为 2.615596

# 逻辑回归建模，计算log_loss
model = LogisticRegression(C=.01)
lrStart = time.time()
model.fit(training[features], training['crime'])
lrCostTime = time.time() - lrStart
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted)
print("逻辑回归建模耗时 %f 秒" % (lrCostTime))
# 逻辑回归建模耗时 50.953264 秒
print("逻辑回归log损失为 %f" % (log_loss(validation['crime'], predicted)))
# 逻辑回归log损失为 2.622638

# 添加犯罪的小时时间点作为特征
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
            'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
            'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

hourFea = [x for x in range(0, 24)]
features = features + hourFea

# 分割训练集(3/5)和测试集(2/5)
training, validation = train_test_split(trainData, train_size=.60)

# 朴素贝叶斯建模，计算log_loss
model = BernoulliNB()
nbStart = time.time()
model.fit(training[features], training['crime'])
nbCostTime = time.time() - nbStart
predicted = np.array(model.predict_proba(validation[features]))
print("朴素贝叶斯建模耗时 %f 秒" % (nbCostTime))
# 朴素贝叶斯建模耗时 0.781367 秒
print("朴素贝叶斯log损失为 %f" % (log_loss(validation['crime'], predicted)))
# 朴素贝叶斯log损失为 2.582740

# 逻辑回归建模，计算log_loss
model = LogisticRegression(C=.01)
lrStart = time.time()
model.fit(training[features], training['crime'])
lrCostTime = time.time() - lrStart
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted)
print("逻辑回归建模耗时 %f 秒" % (lrCostTime))
# 逻辑回归建模耗时 70.921723 秒
print("逻辑回归log损失为 %f" % (log_loss(validation['crime'], predicted)))
# 逻辑回归log损失为 2.592214

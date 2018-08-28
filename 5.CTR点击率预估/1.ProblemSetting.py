'''
## 腾讯移动App广告转化率预估

###题目描述

计算广告是互联网最重要的商业模式之一，广告投放效果通常通过曝光、点击和转化各环节来衡量，大多数广告系统受广告效果数据回流的限制只能通过曝光或点击作为投放效果的衡量标准开展优化。

本题目以移动App广告为研究对象，预测App广告点击后被激活的概率：pCVR=P(conversion=1 | Ad,User,Context)，即给定广告、用户和上下文情况下广告被点击后发生激活的概率。

'''

'''
## 训练数据

### 从腾讯社交广告系统中某一连续两周的日志中按照推广中的App和用户维度随机采样。

每一条训练样本即为一条广告点击日志(点击时间用clickTime表示)，样本label取值0或1，其中0表示点击后没有发生转化，1表示点击后有发生转化，如果label为1，还会提供转化回流时间(conversionTime，定义详见“FAQ”)。

给定特征集如下：
特别的，出于数据安全的考虑，对于userID，appID，特征，以及时间字段，我们不提供原始数据，按照如下方式加密处理：

#### 训练数据文件(train.csv) ####

每行代表一个训练样本，各字段之间由逗号分隔，顺序依次为：“label，clickTime，conversionTime，creativeID，userID，positionID，connectionType，telecomsOperator”。

当label=0时，conversionTime字段为空字符串。

#### 测试数据 ####

从训练数据时段随后1天(即第31天)的广告日志中按照与训练数据同样的采样方式抽取得到，测试数据文件(test.csv)每行代表一个测试样本，各字段之间由逗号分隔，顺序依次为：“instanceID，-1，clickTime，creativeID，userID，positionID，connectionType，telecomsOperator”。其中，instanceID唯一标识一个样本，-1代表label占位使用，表示待预测。

#### 评估方式 ####

通过Logarithmic Loss评估(越小越好)

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

#### 提交格式 ####

模型预估结果以zip压缩文件方式提交，内部文件名是submission.csv。每行代表一个测试样本，第一行为header，可以记录本文件相关关键信息，评测时会忽略，从第二行开始各字段之间由逗号分隔，顺序依次为：“instanceID, prob”，其中，instanceID唯一标识一个测试样本，必须升序排列，prob为模型预估的广告转化概率。

'''

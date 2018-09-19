import pandas as pd
import numpy as np


# 特征工程方法1：histogram
def get_histogram_features(full_dataset):
    def extract_histogram(x):
        count, _ = np.histogram(x, bins=[0, 10, 100, 1000, 10000, 100000, 1000000, 9000000])
        return count

    column_names = ["hist_{}".format(i) for i in range(8)]
    hist = full_dataset.apply(lambda row: pd.Series(extract_histogram(row)), axis=1)
    hist.columns = column_names
    RETURN
    hist


# 特征工程方法2：quantile
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
column_names = ["quantile_{}".format(i) for i in q]
# print pd.DataFrame(train_x)
quantile = pd.DataFrame(x_train).quantile(q=q, axis=1).T
quantile.columns = column_names


# 特征工程方法3：cumsum
def get_cumsum_features(all_features):
    column_names = ["cumsum_{}".format(i) for i in range(len(all_features))]
    cumsum = full_dataset[all_features].cumsum(axis=1)
    cumsum.columns = column_names
    return cumsum


# 特征工程方法4：特征归一化
from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler()
x_train_normal = Scaler.fit_transform(x_train_normal)

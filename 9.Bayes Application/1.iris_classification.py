from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 加载数据
iris = datasets.load_iris()

# 打印前五个四维特征数据
print(iris.data[:5])
'''
    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]
'''

# 假定4个特征独立且服从高斯分布，用贝叶斯分类器建模
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
right_num = (iris.target == y_pred).sum()

# 打印正确率
print("Total testing num :%d , naive bayes accuracy :%f" % (iris.data.shape[0], float(right_num) / iris.data.shape[0]))
# Total testing num :150 , naive bayes accuracy :0.960000

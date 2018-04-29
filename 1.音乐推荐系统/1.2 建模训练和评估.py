'''
    在推荐系统的建模过程中，我们将用到python库Surprise
    (Simple Python RecommendatIon System Engine)是scikit系列中的一个

    简单易用，同时支持多种推荐算法：
        基础算法/baseline algorithms（基于统计产出结果）
        基于近邻方法(协同过滤)/neighborhood methods（基于近邻产出结果）
        矩阵分解方法/matrix factorization-based (SVD, PMF, SVD++, NMF)

    支持不同的评估准则
        rmse 	Compute RMSE (Root Mean Squared Error).
        msd 	Compute MAE (Mean Absolute Error).
        fcp 	Compute FCP (Fraction of Concordant Pairs).
'''
import os

'''
    Jaccard similarity
    交集元素个数/并集元素个数
'''

# 可以使用上面提到的各种推荐系统算法
from surprise import SVD, Reader, GridSearch
from surprise import Dataset
from surprise import evaluate, print_perf

# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')
# k折交叉验证(k=3)
data.split(n_folds=3)
# 试一把SVD矩阵分解
algo = SVD()
# 在数据集上测试一下效果
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
# 输出结果
print_perf(perf)

# 载入自己的数据集方法

# 指定文件所在路径
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating timestamp', sep='\t')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
# 手动切分成5折(方便交叉验证)
data.split(n_folds=5)

'''
    算法调参(让推荐系统有更好的效果)
    这里实现的算法用到的算法无外乎也是SGD等，因此也有一些超参数会影响最后的结果
    我们同样可以用sklearn中常用到的网格搜索交叉验证(GridSearchCV)来选择最优的参数。
'''

# 定义好需要优选的参数网格
# 迭代轮次，学习率，正则程度（超参）
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
# 使用网格搜索交叉验证
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
# 在数据集上找到最好的参数
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)
grid_search.evaluate(data)
# 输出调优的参数组
# 输出最好的RMSE结果
print(grid_search.best_score['RMSE'])
# >>> 0.96117566386

# 输出对应最好的RMSE结果的参数
print(grid_search.best_params['RMSE'])
# >>> {'reg_all': 0.4, 'lr_all': 0.005, 'n_epochs': 10}

# 最好的FCP得分
print(grid_search.best_score['FCP'])
# >>> 0.702279736531

# 对应最高FCP得分的参数
print(grid_search.best_params['FCP'])
# >>> {'reg_all': 0.6, 'lr_all': 0.005, 'n_epochs': 10}

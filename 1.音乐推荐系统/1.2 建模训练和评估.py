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

# 告诉文本阅读器文本的格式
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

'''
    1.用协同过滤构建模型并进行预测movielens的例子
'''

# 可以使用上面提到的各种推荐系统算法
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf

# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')

# k折交叉验证(k=3)
data.split(n_folds=3)

# 试一次协同过滤矩阵分解
algo = KNNWithMeans()

# 在数据集上测试一下效果
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

# 输出结果
print_perf(perf)

"""
以下的程序段告诉大家如何在协同过滤算法建模以后，根据一个item取回相似度最高的item，主要是用到algo.get_neighbors()这个函数
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import io
from surprise import KNNBaseline
from surprise import Dataset


# 获取 电影名->电影id 和 电影id->电影名 的映射
def read_item_names():
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# 打印当前数据格式（稀疏的）：用户id，电影id，打分，时间戳
print(data.raw_ratings[1])

# 首先用算法计算相互间的相似度
data = Dataset.load_builtin('ml-100k')

# 将训练集重新构造矩阵，一个用户对应所有电影维度，打分显示分数，没打分显示0
trainset = data.build_full_trainset()

# 超参数字典：pearson距离，基于item协同过滤
sim_options = {'name': 'pearson_baseline', 'user_based': False}

# 算法选择协同过滤，代入超参数字典
algo = KNNBaseline(sim_options=sim_options)

# 训练集代入，训练过程实质是在计算每一个电影item相似度
algo.train(trainset)

# 获取 电影名->电影id 和 电影id->电影名 的映射
rid_to_name, name_to_rid = read_item_names()

# 查找Toy Story这部电影对应的rid（原始数据id）
toy_story_raw_id = name_to_rid['Toy Story (1995)']
print('Toy Story的raw id：', toy_story_raw_id)
# >>> u'1'

# 查找Toy Story这部电影对应的iid（内部id）
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
print('Toy Story的inner id：', toy_story_inner_id)
# >>> 24

# 找到最近的10个邻居
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
print(toy_story_neighbors)
# >>> 输出10个inner id

# 从近邻的iid->电影名称
# 先由iid->raw id
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
# 再由raw id->电影名称
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)
'''
    >>>
    The 10 nearest neighbors of Toy Story are:
    Beauty and the Beast (1991)
    Raiders of the Lost Ark (1981)
    That Thing You Do! (1996)
    Lion King, The (1994)
    Craft, The (1996)
    Liar Liar (1997)
    Aladdin (1992)
    Cool Hand Luke (1967)
    Winnie the Pooh and the Blustery Day (1968)
    Indiana Jones and the Last Crusade (1989)
'''

'''
    2.音乐预测的例子
'''
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io

from surprise import KNNBaseline, Reader
from surprise import Dataset

import pickle

# 歌单id->歌单名 的映射字典
id_name_dic = pickle.load(open("popular_playlist.pkl", "rb"))
print("加载歌单id到歌单名的映射字典完成...")

# 歌单名->歌单id 的映射字典
name_id_dic = {}
for playlist_id in id_name_dic:
    name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("加载歌单名到歌单id的映射字典完成...")

file_path = os.path.expanduser('./popular_music_suprise_format.txt')

# 指定文件格式：歌单id，歌曲id，打分，时间戳
reader = Reader(line_format='user item rating timestamp', sep=',')

# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)

# 计算歌单和歌单之间的相似度
print("构建数据集...")
trainset = music_data.build_full_trainset()
# sim_options = {'name': 'pearson_baseline', 'user_based': False}

# 随机获取一个歌单名称的key
print(id_name_dic.keys()[2])
# >>> '361197245'

# 用该歌单名称的key查找对应的歌单名称
print(id_name_dic[id_name_dic.keys()[2]])
# >>> 100种深情皆苦 | 你又不知道我难过

# 打印输出训练集items数量
print(trainset.n_items)
# >>> 50539

# 打印输出训练集users数量
print(trainset.n_users)
# >>> 1076

print("开始训练模型...")
# 初始化算法
# sim_options = {'user_based': False}
# algo = KNNBaseline(sim_options=sim_options)
algo = KNNBaseline()

# 代入训练集训练模型
algo.train(trainset)

# 随机获取39号歌单名称，目的找到与39号歌单最近邻的歌单
current_playlist = name_id_dic.keys()[39]
print("歌单名称：", current_playlist)

# 取出近邻
# 映射歌单名称->id
playlist_id = name_id_dic[current_playlist]
print("歌单id：", playlist_id)
# >>> 歌单id：306948578

# 取出来对应的inner user id
playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
print("内部id：", playlist_inner_id)
# >>> 内部id：427

# 找到近邻10个歌单
playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)

# 把歌单id->歌单名称
# inner user id->raw user id
playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
                      for inner_id in playlist_neighbors)
# raw user id->歌单名称
playlist_neighbors = (id_name_dic[playlist_id]
                      for playlist_id in playlist_neighbors)

print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")
for playlist in playlist_neighbors:
    print(playlist, algo.trainset.to_inner_uid(name_id_dic[playlist]))
'''
    >>>
    和歌单 《 世事无常，唯愿你好 》 最接近的10个歌单为：
    【华语】暖心物语 纯白思念 3
    暗暗作祟| 不甘朋友不敢恋人 15
    专属你的周杰伦 18
    「华语歌曲」 23
    [小风收集]21世纪年轻人的音乐 24
    十七岁那年，以为能和你永远 28
    热门流行华语歌曲50首 31
    最易上手吉他弹唱超精选 40
    打开任意门，就有对的人 42
    行车路上，一曲长歌 45
'''

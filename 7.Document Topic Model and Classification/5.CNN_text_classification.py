import jieba
import pandas as pd
import argparse
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics

# 数据预处理
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

# 载入停用词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values


# 构建数据集
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception as e:
            print(line)
            continue


# 生成训练数据
sentences = []

preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')

# 划分训练集测试集
x, y = zip(*sentences)
train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)

"""
    基于卷积神经网络的中文文本分类
"""
learn = tf.contrib.learn

FLAGS = None

# 文档最长长度
MAX_DOCUMENT_LENGTH = 100
# 最小词频数
MIN_WORD_FREQUENCE = 2
# 词嵌入的维度
EMBEDDING_SIZE = 20  # 列特征取的维度数
# filter个数
N_FILTERS = 10
# 感知野大小
WINDOW_SIZE = 20  # 囊括多少个词
# filter的形状
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
# 池化
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0
global n_words


def cnn_model(features, target):
    """2层的卷积神经网络，用于短文本分类"""
    # 生成词嵌入得到一个形状为[n_words, EMBEDDING_SIZE]的词表映射矩阵，用于查找词
    # 接着把一批文本映射成[batch_size, sequence_length, EMBEDDING_SIZE]的三维矩阵形式
    target = tf.one_hot(target, 15, 1, 0)
    word_vectors = tf.contrib.layers.embed_sequence(features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE,
                                                    scope='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # 添加卷积层做滤波
        conv1 = tf.contrib.layers.convolution2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # 添加RELU非线性
        conv1 = tf.nn.relu(conv1)
        # 最大池化
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, POOLING_WINDOW, 1, 1],
            strides=[1, POOLING_STRIDE, 1, 1],
            padding='SAME')
        # 对矩阵进行转置，以满足形状
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # 第2个卷积层
        conv2 = tf.contrib.layers.convolution2d(pool1, N_FILTERS, FILTER_SHAPE2, padding='VALID')
        # 抽取特征
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # 全连接层
    logits = tf.contrib.layers.fully_connected(pool2, 15, activation_fn=None)
    # 交叉熵优化器
    loss = tf.losses.softmax_cross_entropy(target, logits)
    # 优化器选择Adam
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)
    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


# 构建数据集
x_train = pd.DataFrame(train_data)[1]
y_train = pd.Series(train_target)
x_test = pd.DataFrame(test_data)[1]
y_test = pd.Series(test_target)

# VocabularyProcessor词汇预处理模型
tmp = ['I am good', 'you are here', 'I am glad', 'it is great']
# 设定最低补偿维度和最低词频
vocab_processor = learn.preprocessing.VocabularyProcessor(10, min_frequency=1)
print(list(vocab_processor.fit_transform(tmp)))
'''
    [array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
'''

# 处理词汇
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_test = np.array(list(vocab_processor.transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)  # Total words: 50281

cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}
train_target = map(lambda x: cate_dic[x], train_target)
test_target = map(lambda x: cate_dic[x], test_target)
y_train = pd.Series(train_target)
y_test = pd.Series(test_target)

# 构建模型
classifier = learn.SKCompat(learn.Estimator(model_fn=cnn_model))

# 训练和预测
classifier.fit(x_train, y_train, steps=1000)  # 设定迭代次数1000
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))  # Accuracy: 0.888625

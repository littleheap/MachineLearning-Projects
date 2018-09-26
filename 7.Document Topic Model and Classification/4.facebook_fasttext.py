import jieba
import pandas as pd
import random
import fasttext
from gensim.models import Word2Vec

'''
    fasttext是facebook开源的一个词向量与文本分类工具，在学术上没有太多创新点，好处是模型简单，训练速度非常快。
    简单尝试可以发现，用起来还是非常顺手的，做出来的结果也不错，可以达到上线使用的标准。

    简单说来，fastText做的事情，就是把文档中所有词通过lookup table变成向量，取平均后直接用线性分类器得到分类结果。
    论文指出了对一些简单的分类任务，没有必要使用太复杂的网络结构就可以取得差不多的结果。

    fastText论文中提到了两个tricks：
        hierarchical softmax
            类别数较多时，通过构建一个霍夫曼编码树来加速softmax layer的计算，和之前word2vec中的trick相同
        N-gram features
            只用unigram的话会丢掉word order信息，所以通过加入N-gram features进行补充用hashing来减少N-gram的存储
'''

# fastText文本有监督学习

# 数据预处理
# 一共五类文本数据
cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}

# 加载科技文本
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

# 加载汽车文本
df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

# 加载娱乐文本
df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

# 加载军事文本
df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

# 加载运动文本
df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

# 截取部分数据
technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

# 导入停止词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values


# 整理文本数据并标记分类
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append("__label__" + str(category) + " , " + " ".join(segs))
        except Exception as e:
            print(line)
            continue


# 生成训练数据
sentences = []

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

# 打乱数据顺序
random.shuffle(sentences)

# 存储训练数据
print("writing data to fasttext format...")
out = open('train_data.txt', 'w')
for sentence in sentences:
    out.write(sentence.encode('utf8') + "\n")
print("done!")

# 调用fastText训练生成模型
classifier = fasttext.supervised('train_data.txt', 'classifier.model', label_prefix='__label__')

# 评估模型
result = classifier.test('train_data.txt')
print('P@1:', result.precision)  # P@1: 0.972392217757
print('R@1:', result.recall)  # R@1: 0.972392217757
print('Number of examples:', result.nexamples)  # Number of examples: 87584

# 预测实例
label_to_cate = {1: 'technology', 2: 'car', 3: 'entertainment', 4: 'military', 5: 'sports'}

texts = ['中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟 主场 作战 中国队 率先 打破 场上 僵局 利用 角球 机会 大宝 前点 攻门 得手 中国队 领先']
labels = classifier.predict(texts)
print(labels)  # [[u'5']]
print(label_to_cate[int(labels[0][0])])  # sports

labels = classifier.predict_proba(texts)
print(labels)  # [[(u'5', 0.998047)]]

# Top K 个预测结果
labels = classifier.predict(texts, k=3)
print(labels)  # [[u'5', u'2', u'4']]

labels = classifier.predict_proba(texts, k=3)
print(labels)  # [[(u'5', 0.998047), (u'2', 1.95313e-08), (u'4', 1.95313e-08)]]


# fastText文本无监督学习
def preprocess_text_unsupervised(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append(" ".join(segs))
        except Exception as e:
            print(line)
            continue


# 生成无监督训练数据
sentences = []

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

# 存储无监督数据
print("writing data to fasttext unsupervised learning format...")
out = open('unsupervised_train_data.txt', 'w')
for sentence in sentences:
    out.write(sentence.encode('utf8') + "\n")
print("done!")

# Skipgram model
model = fasttext.skipgram('unsupervised_train_data.txt', 'model')
print(model.words)  # list of words in dictionary

# CBOW model
model = fasttext.cbow('unsupervised_train_data.txt', 'model')
print(model.words)  # list of words in dictionary

print(model['赛季'])


# 对比模型
def preprocess_text_unsupervised(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append(" ".join(segs))
        except Exception as e:
            print(line)
            continue


# 生成无监督训练数据
sentences = []

preprocess_text(technology, sentences, cate_dic['technology'])
preprocess_text(car, sentences, cate_dic['car'])
preprocess_text(entertainment, sentences, cate_dic['entertainment'])
preprocess_text(military, sentences, cate_dic['military'])
preprocess_text(sports, sentences, cate_dic['sports'])

model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
model.save("gensim_word2vec.model")
print(model.wv['赛季'])

print(model.wv.most_similar('赛季'))

import jieba
import pandas as pd
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer

'''
    朴素贝叶斯：尝试用朴素贝叶斯完成一个中文文本分类器，在数据量足够，数据丰富度够的情况下，
    用朴素贝叶斯完成这个任务，准确度还可以。
'''

'''
    准备数据:选择科技、汽车、娱乐、军事、运动5类文本数据进行处理
'''
# 科技文本
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

# 汽车文本
df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

# 娱乐文本
df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

# 军事文本
df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

# 运动文本
df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

# 截取部分数据
technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

print(technology[12], end='\n')
'''
    现在家里都拉了网线，都能无线上网，一定要帮他们先登上WiFi，另外，老人不懂得流量是什么，也不知道如何开关，控制流量，所以设置好流量上限很重要，免得不小心点开了视频或者下载，电话费就大发了。
'''

print(car[100], end='\n')
'''
    截至发稿时，人人车给出的处理方案仍旧是检修车辆。王先生则认为，车辆在购买时就存在问题，但交易平台并未能检测出来。因此，王先生希望对方退款。王先生称，他将找专业机构对车辆进行鉴定，并通过法律途径维护自己的权益。J256
'''

print(entertainment[10], end='\n')
'''
    网综尺度相对较大原本是制作优势，《奇葩说》也曾经因为讨论的话题较为前卫一度引发争议。但《奇葩说》对于价值观的把握和引导让其中内含的“少儿不宜”只能算是小花絮。而纯粹是为了制造话题而“污”得“无节操无下限”的网综不仅让人生厌，也给节目自身招致了下架的厄运。对资本方而言，即便只从商业运营考量，点击量也分有价值和无价值，节目内容的变现能力如果建立在博眼球和低趣味迎合上，商业运营也难长久。对节目制作方与平台来说，为博一时的高点击而不顾底线不仅是砸自己招牌，以噱头吸引而来的观众与流量也是难以维持。
'''

print(military[10], end='\n')
'''
    央视记者 胡善敏：我现在所处的位置是在辽宁舰的飞行甲板，执行跨海区训练和试验任务的辽宁舰官兵，正在展开多个科目的训练，穿着不同颜色服装的官兵在紧张的对舰载机进行转运。
'''

print(sports[10], end='\n')
'''
    据统计，2016年仅在中国田径协会注册的马拉松赛事便达到了328场，继续呈现出爆发式增长的态势，2015年，这个数字还仅仅停留在134场。如果算上未在中国田协注册的纯“民间”赛事，国内全年的路跑赛事还要更多。
'''

# 载入停止词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')

stopwords = stopwords['stopword'].values


# 去停止词
def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            # 删除空字符
            segs = filter(lambda x: len(x) > 1, segs)
            # 删除停止词
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

# 打乱数据顺序
random.shuffle(sentences)

# 输出部分分词数据和对应分类标记
for sentence in sentences[:10]:
    print(sentence[0], sentence[1])
'''
    金山毒霸 专家建议 technology
    解释 战友 很快 明白 航母 机电 空调 系统 重要性 阮万林 竖起 大拇指 military
    实施 共享 停车 问题重重 car
    中国区 独立 郑杰 权力 职能 菲克 拓展 中国 市场 合资 公司 全球化 益处 中国 市场 创新 影响 全球 市场 郑杰 中国 消费者 需求 中国 需求 全球 需求 中国 市场 全球 第一 市场 汽车 公司 中国 本土化 中国 改进 产品 这是 意识 中国 全球 地位 car
    关注 爆炸 原因 电池 缺陷 technology
    首发 命中 录像 回放 一旁 领队 黄庆利 激动 好家伙 发现 摧毁 露头 military
    VV7c VV7s 配备 2.0 双流 涡轮 增压 缸内 直喷 发动机 车速 205km 额定功率 172kw 扭矩 360 2200 4000 rps 转速 扭矩 带来 加速性 兼顾 燃油 经济性 车辆 采用 湿式 离合 变速器 相比 传统 7AT 提升 燃油 经济性 减少 油耗 car
    TechCrunch 人工智能 人才 角度 肯定 陆奇 加盟 文章 指出 百度 人工智能 人才 招募 发力 任命 微软 高管 知名 AI 专家 陆奇 集团 总裁兼 COO 百度 搜索 闻名 中国 市场 占据 主导地位 近两年来 百度 集中精力 发展 包括 无人 在内 人工智能 陆奇 加盟 成功 technology
    河尾滩 边防连 巡逻 分队 天文 边防连 巡逻 分队 克服 困难 上级 指定 时间 指定 地点 会合 平时 距离 不远 难得一见 战友 神圣 边境线 相聚 那种 激动 那种 喜庆 堪比 过年 会哨 两支 巡逻 分队 返回 哨所 夕阳西下 一轮 硕大 明月 跃出 连绵 雪峰 military
    宝马 i5 车型 适合 家庭 新车型 保留 i3 车型 自杀式 车门 前卫 造型 更大 实用 车身 强大 续航力 特斯拉 Model 车型 目标 报道 显示 宝马 放弃 推出 i5 车型 报道 属实 宝马 系列 车型 在短期内 只会 推出 i8 Spyder 车型 宝马 公司 2021 推出 一款 名为 iNext 自动 驾驶 汽车 car
'''

# 分割训练集的测试集
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

print(len(x_train))  # 65696
print(len(x_test))  # 21899

# 制作词袋模型（构建词索引字典）
vec = CountVectorizer(
    analyzer='word',  # 基于词而不是n-gram滑窗
    max_features=4000,  # 取最高频4000词
)

# fit之后就算是拿到一个词-向量的映射器
vec.fit(x_train)

# 然后可以用vec对已有的x目标词汇进行向量特征映射
def get_features(x):
    vec.transform(x)

# 导入模型
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)

# 查看准确率
print(classifier.score(vec.transform(x_test), y_test))  # 0.83474131238869353

# 尝试提高准确率，加入抽取2-gram和3-gram的统计特征
vec = CountVectorizer(
    analyzer='word',  # 基于词
    ngram_range=(1, 4),  # 规定n-gram范围，获取更健壮的特征
    max_features=20000,  # 取前20000高频词
)
vec.fit(x_train)


def get_features(x):
    vec.transform(x)


# 分类训练
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)

# 查看准确率
print(classifier.score(vec.transform(x_test), y_test))  # 0.87401251198684871


# 引入交叉验证
def stratifiedkfold_cv(x, y, clf_class, shuffle=True, n_folds=5, **kwargs):
    # 依据y标签生成K折数据
    stratifiedk_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y[:]
    # 遍历每一份数据，4折做训练，1折做验证
    for train_index, test_index in stratifiedk_fold:
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


NB = MultinomialNB
print(precision_score(y, stratifiedkfold_cv(vec.transform(x), np.array(y), NB), average='macro'))  # 0.88154693235


# 自定义分类器
class TextClassifier():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 4), max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)

print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))  # ['military']
print(text_classifier.score(x_test, y_test))  # 0.865427645098

# SVM文本分类
svm = SVC()
svm.fit(vec.transform(x_train), y_train)
print(svm.score(vec.transform(x_test), y_test))

# 尝试其他特征和模型
class TextClassifier():

    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=12000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)

print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))  # ['military']
print(text_classifier.score(x_test, y_test))  # 0.874788803142

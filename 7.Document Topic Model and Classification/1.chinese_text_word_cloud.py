import warnings
import jieba
import numpy
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator

warnings.filterwarnings("ignore")

'''
    中文词云可视化
'''

# 读取数据
df = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
print(df.head())
'''
       Unnamed: 0                                            content
    0           0  　　2016年是综艺井喷的一年，《2016年中国网络视听发展研究报告》数据显示，截至2016...
    1           1                                               　　区别
    2           2                                 　　平台决定了资源的多寡和资本的投入
    3           3  　　网络综艺和电视综艺最大的区别在哪？其实是平台。因为平台决定了资源的多寡和资本的投入。所以...
    4           4  　　网络综艺与电视综艺在播出模式、观众群体以及节目板块等方面也都存在差异。在传播上，电视台比...
'''

# 将空行过滤掉
df = df.dropna()
# content转成列表
content = df.content.values.tolist()
segment = []
for line in content:
    try:
        # 分词处理
        segs = jieba.lcut(line)
        for seg in segs:
            if len(seg) > 1 and seg != '\r\n':
                # 将每个分词放到segment中
                segment.append(seg)
    except:
        print(line)
        continue

# 删除停止词操作
words_df = pd.DataFrame({'segment': segment})
print(words_df.head())
'''
      segment
    0    2016
    1      综艺
    2      井喷
    3      一年
    4    2016
'''
# 读取停止词词典
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
print(stopwords.head())
'''
      stopword
    0        !
    1        "
    2        #
    3        $
    4        %
'''

# 删除停止词
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]

# 统计词频
words_stat = words_df.groupby(by=['segment'])['segment'].agg({"计数": numpy.size})
words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)
print(words_stat.head())
'''
          segment     计数
    60811      电影  10230
    73265      观众   5574
    8615       中国   5476
    70481      节目   4398
    33623      导演   4197
'''

# 制作词云
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
wordcloud = WordCloud(font_path="data/simhei.ttf", background_color="white", max_font_size=80)
# 只取前1000个词，生成模范字典，key是词，value是频次
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()

# 自定义背景生成词云
matplotlib.rcParams['figure.figsize'] = (15.0, 15.0)
# 读取背景图
bimg = imread('figures/entertainment.jpeg')
# 背景色是白色
wordcloud = WordCloud(background_color="white", mask=bimg, font_path='data/simhei.ttf', max_font_size=200)
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
bimgColors = ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()

# 体育新闻
df = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df = df.dropna()
content = df.content.values.tolist()
segment = []
for line in content:
    try:
        segs = jieba.lcut(line)
        for seg in segs:
            if len(seg) > 1 and seg != '\r\n':
                segment.append(seg)
    except:
        print(line)
        continue

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
words_df = pd.DataFrame({'segment': segment})
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]
words_stat = words_df.groupby(by=['segment'])['segment'].agg({"计数": numpy.size})
words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)

wordcloud = WordCloud(font_path="data/simhei.ttf", background_color="black", max_font_size=80)
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()

# 自定义背景
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
bimg = imread('figures/sports.jpeg')
wordcloud = WordCloud(background_color="white", mask=bimg, font_path='data/simhei.ttf', max_font_size=200)
word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
wordcloud = wordcloud.fit_words(word_frequence)
bimgColors = ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()

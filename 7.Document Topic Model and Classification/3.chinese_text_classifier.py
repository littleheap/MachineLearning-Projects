import jieba
import pandas as pd
import random

'''
    朴素贝叶斯:我们试试用朴素贝叶斯完成一个中文文本分类器，一般在数据量足够，数据丰富度够的情况下，用朴素贝叶斯完成这个任务，准确度还是很不错的。
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

# 打乱数据
random.shuffle(sentences)

# 输出部分数据
for sentence in sentences[:10]:
    print(sentence[0], sentence[1])

from gensim import corpora, models, similarities
import gensim
import jieba
import jieba.analyse as analyse
import pandas as pd

'''
    关键词提取：基于TF-IDF算法（基于频度+全局抵消）
    基本思想：如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，
    则认为此词或者短语具有很好的类别区分能力，适合用来分类。
    TFIDF实际上是：TF * IDF，TF词频(Term Frequency)，IDF逆向文件频率(Inverse Document Frequency)
    
    jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
        ·sentence：待提取的文本
        ·topK：返回几个 TF/IDF 权重最大的关键词，默认值为 20
        ·withWeight：是否返回关键词权重值，默认值为 False
        ·allowPOS：仅包括指定词性的词，默认值为空，即不筛选
'''
# 科技新闻主题提取
df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()
# 拼接文本
content = "".join(lines)
# 使用jieba计算TF-IDF，选取权重Top20
print("  ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))
'''
用户  2016  互联网  手机  平台  人工智能  百度  2017  智能  技术  数据  360  服务  直播  产品  企业  安全  视频  移动  应用  网络  行业  游戏  机器人  电商  内容  中国  领域  通过  发展
'''

# 军事新闻主题提取
df = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()
content = "".join(lines)
# 使用jieba计算TF-IDF，选取权重Top20
print("  ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))
'''
航母  训练  海军  中国  官兵  部队  编队  作战  10  任务  美国  导弹  能力  20  2016  军事  无人机  装备  进行  记者  我们  军队  安全  保障  12  战略  军人  日本  南海  战机
'''

'''
    关键词抽取：基于TextRank算法
    jieba.analyse.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
    默认过滤词性
    jieba.analyse.TextRank() 
        新建自定义TextRank 实例
    算法论文：《TextRank: Bringing Order into Texts》
    基本思想:
        ·将待抽取关键词的文本进行分词
        ·以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
        ·计算图中节点的PageRank得到重要性，注意是无向带权图
'''
df = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()
content = "".join(lines)
# 包括动词
print("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
print("----------------分割线----------------")
# 排除动词
print("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n'))))
'''
中国  海军  训练  美国  部队  进行  官兵  航母  作战  任务  能力  军事  发展  工作  国家  问题  建设  导弹  编队  记者
----------------分割线----------------
中国  海军  美国  部队  官兵  航母  军事  国家  任务  能力  导弹  技术  问题  日本  军队  编队  装备  系统  记者  战略
'''

'''
    LDA主题模型：
    首先把文本内容处理成固定的格式，一个包含句子的list，list中每个元素是分词后的词list。
    实例：[[第，一，条，新闻，在，这里],[第，二，条，新闻，在，这里],[这，是，在，做， 什么],...]
'''
# 载入暂停词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values

# 转换文本格式
df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines = df.content.values.tolist()

sentences = []
for line in lines:
    try:
        segs = jieba.lcut(line)
        # 删除空词
        segs = filter(lambda x: len(x) > 1, segs)
        # 删除停止词
        segs = filter(lambda x: x not in stopwords, segs)
        sentences.append(segs)
    except Exception as e:
        print(line)
        continue

# 查看格式，打印第六句话每一个词
for word in sentences[5]:
    print(word)
'''
本次 商汤 带来 黄仁勋 展示 遥相呼应 SenseFace 人脸 布控 系统 千万级 人员 库中 300ms 识别 瞬间 锁定目标 功耗 十几 当属 人脸 布控 一大 科技
'''

# 词袋模型
dictionary = corpora.Dictionary(sentences)
# 词索引映射建立
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
# 打印第五句话每个词的索引
print(corpus[5])
'''
    [(21, 1),
     (25, 1),
     (54, 1),
     (59, 1),
     (79, 1),
     (80, 1),
     (81, 1),
     (91, 1),
     (103, 1),
     (104, 2),
     (105, 2),
     (112, 1),
     (126, 1),
     (130, 1),
     (131, 1),
     (132, 1),
     (133, 1),
     (134, 1),
     (135, 1),
     (136, 1),
     (137, 1),
     (138, 1)]
'''

# LDA建模
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# 查看模型中第3号分类，Top5个词
print(lda.print_topic(3, topn=5))
# 0.040*"产品" + 0.016*"品牌" + 0.016*"消费者" + 0.015*"市场" + 0.012*"体验"

# 打印所有主题，显示Top8个词
for topic in lda.print_topics(num_topics=20, num_words=8):
    print(topic[1])
'''
    0.024*"发展" + 0.020*"企业" + 0.017*"技术" + 0.015*"产业" + 0.014*"中国" + 0.014*"创新" + 0.013*"行业" + 0.013*"领域"
    0.020*"直播" + 0.011*"活动" + 0.009*"国美" + 0.008*"母婴" + 0.007*"现场" + 0.006*"比特" + 0.006*"电视" + 0.006*"生活"
    0.014*"软件" + 0.010*"云端" + 0.010*"时间" + 0.008*"文件" + 0.008*"小时" + 0.008*"隔离" + 0.006*"北京" + 0.006*"实时"
    0.040*"产品" + 0.016*"品牌" + 0.016*"消费者" + 0.015*"市场" + 0.012*"体验" + 0.008*"用户" + 0.008*"消费" + 0.007*"学校"
    0.033*"用户" + 0.019*"勒索" + 0.018*"信息" + 0.018*"手机" + 0.016*"攻击" + 0.016*"网络" + 0.012*"系统" + 0.011*"诈骗"
    0.034*"智能" + 0.019*"数据" + 0.018*"技术" + 0.014*"互联网" + 0.013*"服务" + 0.011*"企业" + 0.011*"提供" + 0.010*"平台"
    0.023*"增长" + 0.021*"公司" + 0.017*"亿元" + 0.015*"业务" + 0.015*"孩子" + 0.015*"收入" + 0.013*"家长" + 0.012*"同比"
    0.018*"流量" + 0.011*"微信" + 0.010*"高通" + 0.010*"知识产权" + 0.007*"蓝色" + 0.007*"量子" + 0.006*"费用" + 0.006*"4G"
    0.061*"百度" + 0.043*"人工智能" + 0.025*"技术" + 0.022*"VR" + 0.012*"学习" + 0.010*"永恒" + 0.008*"机器" + 0.007*"识别"
    0.024*"数据" + 0.015*"宽带" + 0.007*"防御" + 0.007*"测试" + 0.007*"物流" + 0.007*"提速" + 0.006*"公司" + 0.006*"资金"
    0.012*"摄像头" + 0.010*"美团" + 0.010*"外卖" + 0.008*"点评" + 0.008*"妈妈" + 0.007*"星际" + 0.007*"拍照" + 0.006*"课堂"
    0.018*"用户" + 0.018*"数据" + 0.013*"联想" + 0.011*"杨元庆" + 0.011*"老师" + 0.010*"医疗" + 0.009*"医生" + 0.008*"搜狗"
    0.043*"360" + 0.024*"漏洞" + 0.020*"手机" + 0.009*"网站" + 0.008*"QQ" + 0.008*"应急" + 0.006*"搜索" + 0.006*"修复"
    0.019*"电商" + 0.013*"共享" + 0.010*"平台" + 0.009*"城市" + 0.008*"政务" + 0.007*"数据" + 0.007*"数据中心" + 0.007*"报告"
    0.010*"信息安全" + 0.008*"地图" + 0.007*"人类" + 0.006*"人工智能" + 0.006*"员工" + 0.006*"实验室" + 0.005*"人才" + 0.005*"力量"
    0.087*"游戏" + 0.012*"玩家" + 0.010*"独立" + 0.007*"联盟" + 0.006*"学生" + 0.006*"杭州" + 0.005*"协议" + 0.005*"平台"
    0.022*"中国" + 0.016*"数据" + 0.014*"市场" + 0.013*"城市" + 0.008*"发展" + 0.008*"旅游" + 0.008*"建设" + 0.008*"战略"
    0.026*"中国" + 0.020*"腾讯" + 0.017*"创业" + 0.011*"2017" + 0.009*"全球" + 0.008*"日电" + 0.008*"电竞" + 0.008*"中新网"
    0.052*"手机" + 0.013*"市场" + 0.012*"苹果" + 0.011*"小米" + 0.010*"智能手机" + 0.009*"联想" + 0.008*"金立" + 0.008*"无人机"
    0.034*"内容" + 0.028*"平台" + 0.027*"视频" + 0.024*"用户" + 0.019*"病毒" + 0.016*"直播" + 0.012*"营销" + 0.009*"广告"
'''

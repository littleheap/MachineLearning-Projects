import multiprocessing
import gensim
import sys
from random import shuffle

'''
    word2vec到song2vec：
    我们把歌曲的id序列取出来，类比于分完词后的句子，送到word2vec中去学习一下，看看效果
'''

'''
    Step 1：准备歌曲序列+训练嵌入模型
'''


# 准备歌曲序列数据
def parse_playlist_get_sequence(in_line, playlist_sequence):
    song_sequence = []
    contents = in_line.strip().split("\t")
    # 解析歌单序列
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_sequence.append(song_id)
        except:
            print("song format error")
            print(song + "\n")
    for i in range(len(song_sequence)):
        # 将歌单内的歌曲序列不断打乱
        shuffle(song_sequence)
        # 不断打乱不断添加到sequence
        playlist_sequence.append(song_sequence)


# 模型训练函数
def train_song2vec(in_file, out_file):
    # 所有歌单序列
    playlist_sequence = []
    # 遍历所有歌单获得歌曲序列
    for line in open(in_file):
        parse_playlist_get_sequence(line, playlist_sequence)
    # 使用word2vec训练
    cores = multiprocessing.cpu_count()
    print("using all " + str(cores) + " cores")
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150, min_count=3, window=7, workers=cores)
    print("Saving model...")
    model.save(out_file)


song_sequence_file = "./popular.playlist"
model_file = "./song2vec.model"
train_song2vec(song_sequence_file, model_file)

'''
    Step 2：测试模型进行预测
'''

import pickle

# 歌曲字典数据
song_dic = pickle.load(open("popular_song.pkl", "rb"))
# 模型载入
model_str = "./song2vec.model"
model = gensim.models.Word2Vec.load(model_str)

for song in song_dic.keys()[:10]:
    print(song, song_dic[song])
'''
    287140 梦不落	孙燕姿
    445845011 狂想.Rhapsody	冯建宇
    110557 灰色空间	罗志祥
    10308003 偏偏喜欢你	陈百强
    28029940 拥抱的理由	尹熙水
    28029946 三个人的错	王菀之
    28029947 拥抱的理由	李泰
    27591219 拍错拖	卫兰
    28029949 我是你的谁	张含韵
    31134863 没有用	徐誉滕
'''

song_id_list = song_dic.keys()[1000:1500:50]
for song_id in song_id_list:
    result_song_list = model.most_similar(song_id)
    print(song_id, song_dic[song_id])
    print("\n相似歌曲 和 相似度 分别为:")
    for song in result_song_list:
        print("\t", song_dic[song[0]], song[1])
    print("\n")
'''
415085693 【钢琴】皈依	昼夜

相似歌曲                                                         相似度 :

刚好遇见你	昼夜                                                 0.985483407974
【10p】命运 ——网游小说《猎者天下》群像歌曲（Cover Memoria）	人衣大人  0.982731580734
世末歌者（Cover：乐正凌）	萧忆情Alex                                0.967274367809
【钢琴】雨落长安	昼夜                                             0.96506023407
是你	灰白                                                         0.921878576279
南部小城	李蚊香                                                   0.899908363819
【钢琴】棠梨煎雪	昼夜                                             0.89453458786
 一剑逍遥	小义学长                                              0.892150878906
【钢琴】dying in the sun	昼夜                                     0.884359836578
【钢琴】团子大家族	昼夜                                         0.827963769436
'''

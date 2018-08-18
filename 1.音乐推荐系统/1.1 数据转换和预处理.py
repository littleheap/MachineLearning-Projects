import json
import sys

'''
    假设：同一个歌单中的歌曲，有比较高的相似性；同时都是做单的人喜欢的
    Step 1：原始数据->歌单数据+歌曲数据：
        抽取：歌单名称、歌单id、收藏数、所属分类——4个歌单维度的信息
        抽取：歌曲id、歌曲名、歌手、歌曲热度——4个维度信息歌曲的信息
        进行拼接
        
    效果：[歌单名称##歌单分类##歌单id##歌单收藏数    歌曲id:::歌曲名称:::歌手:::歌曲热度    歌曲id:::歌曲名称:::歌手:::歌曲热度 ...]
    eg:
    漫步西欧小镇上##小语种,旅行##69413685##474    18682332:::Wäg vo dir::Joy Amelie::70.0    4335372::Only When I Sleep::The Corrs::60.0    2925502::Si Seulement::Lynnsha::100.0    21014930::Tu N'As Pas Cherché...::La Grande Sophie::100.0    20932638::Du behöver aldrig mer vara rädd::Lasse Lindh::25.0    17100518::Silent Machine::Cat Power::60.0    3308096::Kor pai kon diew : ชอไปคนเดียว::Palmy::5.0    1648250::les choristes::Petits Chanteurs De Saint Marc::100.0    4376212::Paddy's Green Shamrock Shore::The High Kings::25.0    2925400::A Todo Color::Las Escarlatinas::95.0    19711402::Comme Toi::Vox Angeli::75.0    3977526::Stay::Blue Cafe::100.0    2538518::Shake::Elize::85.0    2866799::Mon Ange::Jena Lee::85.0    5191949::Je M'appelle Helene::Hélène Rolles::85.0    20036323::Ich Lieb' Dich Immer Noch So Sehr::Kate & Ben::100.0
'''


# 解析每一条歌单JSON数据函数
def parse_song_line(in_line):
    # 读取JSON歌单数据
    data = json.loads(in_line)
    # 获取歌单名称
    name = data['result']['name']
    # 获取歌单类别
    tags = ",".join(data['result']['tags'])
    # 获取收藏数
    subscribed_count = data['result']['subscribedCount']
    # 过滤：如果收藏数小于100
    if (subscribed_count < 100):
        return False
    # 获取歌单id
    playlist_id = data['result']['id']
    # 歌曲数据信息
    song_info = ''
    # 获取歌曲字典列表
    songs = data['result']['tracks']
    # 遍历所有歌曲
    for song in songs:
        # 用':::'间隔拼接歌曲信息，包括：id、歌名、歌手、热度
        try:
            song_info += "\t" + ":::".join(
                [str(song['id']), song['name'], song['artists'][0]['name'], str(song['popularity'])])
        except Exception as e:
            continue
    # 返回整合后的歌单和对应系列歌曲的信息
    return name + "##" + tags + "##" + str(playlist_id) + "##" + str(subscribed_count) + song_info


# 解析文件函数
def parse_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file):
        result = parse_song_line(line)
        if (result):
            out.write(result.encode('utf-8').strip() + "\n")
    out.close()


# parse_file("./playlist_detail_all.json", "./163_music_playlist.txt")

'''
    开发原则：
        project = offline modelling + online prediction
        1）offline：python脚本语言
        2）online：效率至上 C++/Java
        原则：能离线预先算好的，都离线算好，最优的形式：线上是一个K-V字典
'''

'''
    两种推荐方案：
    1.针对用户推荐：网易云音乐(每日30首歌/7首歌)
    2.针对歌曲：在你听某首歌的时候，找“相似歌曲”
'''

'''
    Step 2：歌单+歌曲数据->推荐系统格式数据：
        主流的python推荐系统框架，支持的最基本数据格式为movielens dataset
        其评分fit数据格式为 “user item rating timestamp” —— “歌单id、歌曲id、热度、时间戳”
        为了简单，也把数据处理成这个格式
'''

import surprise
import lightfm
import json
import sys


def is_null(s):
    return len(s.split(",")) > 2


# 解析歌曲info信息
def parse_song_info(song_info):
    try:
        # 获取每首歌的四个特征
        song_id, name, artist, popularity = song_info.split(":::")
        # 打分默认为1.0，时间戳1300000
        return ",".join([song_id, "1.0", '1300000'])
    except Exception as e:
        return ""


# 解析歌单info信息
def parse_playlist_line(in_line):
    try:
        # 分离歌单信息，歌曲信息
        contents = in_line.strip().split("\t")
        # 解析歌单信息：歌单名称、所属分类、歌单id、收藏数
        name, tags, playlist_id, subscribed_count = contents[0].split("##")
        # 解析歌曲信息：歌曲所在歌单id、歌曲id，打分，时间戳
        songs_info = map(lambda x: playlist_id + "," + parse_song_info(x), contents[1:])
        # 过滤信息
        songs_info = filter(is_null, songs_info)
        # 返回歌曲信息
        return "\n".join(songs_info)
    except Exception as e:
        print(e)
        return False


def parse_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file):
        result = parse_playlist_line(line)
        if (result):
            out.write(result.encode('utf-8').strip() + "\n")
    out.close()


# parse_file("./163_music_playlist.txt", "./163_music_suprise_format.txt")
# parse_file("./popular.playlist", "./popular_music_suprise_format.txt")

'''
    Step 3：保存歌单和歌曲信息备用：
    需要保存 歌单id=>歌单名 和 歌曲id=>歌曲名 的信息后期备用。
'''

import pickle as pickle
import sys


# 生成歌单和歌曲id字典函数
def parse_playlist_get_info(in_line, playlist_dic, song_dic):
    # 分离歌单信息，歌曲信息
    contents = in_line.strip().split("\t")
    # 获取歌单名、类别、歌单id、订阅数
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    # 歌单字典建立
    playlist_dic[playlist_id] = name
    # 遍历所有歌曲
    for song in contents[1:]:
        try:
            # 获取歌曲信息
            song_id, song_name, artist, popularity = song.split(":::")
            # 歌曲字典建立
            song_dic[song_id] = song_name + "\t" + artist
        except:
            print("song format error")
            print(song + "\n")


def parse_file(in_file, out_playlist, out_song):
    # 从歌单id到歌单名称的映射字典
    playlist_dic = {}
    # 从歌曲id到歌曲名称的映射字典
    song_dic = {}
    for line in open(in_file):
        parse_playlist_get_info(line, playlist_dic, song_dic)
    # 把映射字典保存在二进制文件中
    pickle.dump(playlist_dic, open(out_playlist, "wb"))
    # 可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
    pickle.dump(song_dic, open(out_song, "wb"))

# parse_file("./163_music_playlist.txt", "playlist.pkl", "song.pkl")
# parse_file("./popular.playlist", "popular_playlist.pkl", "popular_song.pkl")

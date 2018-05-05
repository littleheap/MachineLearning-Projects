import operator
import re

import funs

# 复杂计算两个词竞争值
# print(funs.comp('奔驰', '奥迪'))  # >>> 0.019
# print(funs.comp('华为', '小米'))  # >>> 0.038
# print(funs.comp('华为', '三星'))  # >>> 0.029
# print(funs.comp('ipad', 'iphone'))  # >>> 0.008
# print(funs.comp('锤子', '小米'))  # >>> 0.016
# print(funs.comp('百度', '腾讯'))  # >>> 0.008
# print(funs.comp('男生', '女生'))  # >>> 0.165

# 优化计算两个词竞争值
# print(funs.comp_simple(funs.getdic('奔驰'), '奔驰', '宝马'))

# 获得一个词的相关词词典
# print(funs.getdic('奔驰'))

# 计算两个词的相关词词典长度
# print(funs.intersects_length(funs.getdic('奔驰'), '宝马'))

'''
    开始正式程序
'''

def run(seed):
    # 设定种子词
    # seed = '奔驰'

    # 读取基础词汇数据
    read = open('./data/NoneWord.1000.Change.TRAIN', 'r', encoding='gbk', errors='ignore')

    data = read.read()

    rows = data.split('\n')

    # 初始化匹配词集合和字典
    match_set = set()

    match_dic = {}

    i = 0

    # 初始获取种子词相关词词典
    s_dic = funs.getdic(seed)

    k = 1

    for word in rows:
        i = i + 1
        # 过滤非中文词汇
        word = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", word)
        if len(word) == 0:
            continue
        if operator.eq(word, ''):
            continue
        if len(word) == 1:
            continue
        if funs.intersects_length(s_dic, word) < 5:
            print('%.2f' % (i / 1200 * 100), '%')
            continue
        score = funs.comp_simple(s_dic, seed, word)
        if len(match_dic) < 20:
            match_dic[word] = score
        else:
            for key in list(match_dic.keys()):
                if score > match_dic[key]:
                    del match_dic[key]
                    match_dic[word] = score
                    break
        print('%.2f' % (i / 1200 * 100), '%')

    read.close()

    print('----------final----------')

    # 排序竞争词
    sort = sorted(match_dic.items(), key=lambda d: d[1], reverse=True)

    print(sort)

    for item in sort:
        print(item)

run('华为')
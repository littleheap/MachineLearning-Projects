import operator
import jieba.posseg

'''
    计算s和k竞争值，基于14W搜索词条
'''


def comp(s, k):
    # 读取数据
    f = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

    data = f.read()

    rows = data.split('\n')

    print('---------', 'seed =', s, '  ', 'k =', k, '---------')

    # 统计seed相关词词典
    num_seed = 0

    dic_seed = set()

    # seed
    for row in rows:
        if row.find(s) != -1:
            segs = jieba.posseg.cut(row)
            num_seed = num_seed + 1
            for seg in segs:
                dic_seed.add(seg.word)

    # 将seed从自己从相关字典中移除
    dic_seed.remove(s)

    print('seed =', s, ' 相关词词典：', dic_seed)
    print('seed =', s, ' 相关词词典长度：', len(dic_seed))

    # 统计k相关词词典
    num_k = 0

    dic_k = set()

    # k
    for row in rows:
        if row.find(k) != -1:
            segs = jieba.posseg.cut(row)
            num_k = num_k + 1
            for seg in segs:
                dic_k.add(seg.word)

    # 将k从自己从相关字典中移除
    dic_k.remove(k)

    print('k =', k, ' 相关词词典：', dic_k)
    print('k =', k, ' 相关词词典长度：', len(dic_k))

    # s和k相关词交集
    dic_seed_k = dic_seed & dic_k

    print('seed =', s, ' k=', k, ' 相关词交集：', dic_seed_k)
    print('seed =', s, ' k=', k, ' 相关词交集长度：', len(dic_seed_k))

    # 竞争性计算
    print('---------竞争性计算----------')

    comp = 0

    # 计算 S
    i_s = 0

    for row in rows:
        if row.find(s) != -1:
            i_s = i_s + 1

    # 遍历中介词A
    proc = 0
    for a in dic_seed_k:
        if operator.eq(a, ''):
            proc = proc + 1
            continue
        proc = proc + 1
        # 计算 A
        i_a = 0
        for row in rows:
            if row.find(a) != -1:
                i_a = i_a + 1
        # 计算 SA
        i_sa = 0
        for row in rows:
            if row.find(a) != -1:
                if row.find(s) != -1:
                    i_sa = i_sa + 1
        # 计算 KA
        i_ka = 0
        for row in rows:
            if row.find(a) != -1:
                if row.find(k) != -1:
                    i_ka = i_ka + 1
        # 计算 WA
        wa = i_sa / i_s
        # 累计计算 COMP
        comp = comp + wa * i_ka / (i_a - i_sa)
        # 打印完成百分比
        print(proc / len(dic_seed_k) * 100, '%')

    print('seed =', s, ' k =', k, ' 竞争性：', comp)

    return comp


def comp_simple(s_dic, s, k):
    # 读取数据
    f = open('140000.TRAIN', 'r', encoding='gbk', errors='ignore')
    data = f.read()
    rows = data.split('\n')

    # print('---------', 'seed =', s, '  ', 'k =', k, '---------')

    '''
    # seed相关词词典
    num_seed = 0

    dic_seed = set()

    # seed
    for row in rows:
        if row.find(s) != -1:
            segs = jieba.posseg.cut(row)
            num_seed = num_seed + 1
            for seg in segs:
                dic_seed.add(seg.word)

    dic_seed.remove(s)
    '''
    dic_seed = s_dic
    # print('seed =', s, ' 相关词词典：', dic_seed)
    # print('seed =', s, ' 相关词词典长度：', len(dic_seed))

    # k相关词词典
    num_k = 0

    dic_k = set()

    # k
    for row in rows:
        if row.find(k) != -1:
            segs = jieba.posseg.cut(row)
            num_k = num_k + 1
            for seg in segs:
                dic_k.add(seg.word)

    dic_k.remove(k)
    # print('k =', k, ' 相关词词典：', dic_k)
    # print('k =', k, ' 相关词词典长度：', len(dic_k))

    # 奔驰奥迪相关性交集
    dic_seed_k = dic_seed & dic_k

    if len(dic_seed_k) == 0:
        return 0

    # print('seed =', s, ' k=', k, ' 相关词交集：', dic_seed_k)
    # print('seed =', s, ' k=', k, ' 相关词交集长度：', len(dic_seed_k))

    # 竞争性计算
    # print('---------竞争性计算----------')

    comp = 0

    # 计算 S
    i_s = 0

    for row in rows:
        if row.find(s) != -1:
            i_s = i_s + 1

    # 遍历中介词
    proc = 0
    for a in dic_seed_k:
        if operator.eq(a, ''):
            proc = proc + 1
            continue
        proc = proc + 1
        # 计算 A
        i_a = 0
        for row in rows:
            if row.find(a) != -1:
                i_a = i_a + 1
        # 计算 SA
        i_sa = 0
        for row in rows:
            if row.find(a) != -1:
                if row.find(s) != -1:
                    i_sa = i_sa + 1
        # 计算 KA
        i_ka = 0
        for row in rows:
            if row.find(a) != -1:
                if row.find(k) != -1:
                    i_ka = i_ka + 1
        # 计算 WA
        wa = i_sa / i_s
        # 累计计算 COMP
        if i_a - i_sa == 0:
            return 0
        comp = comp + wa * i_ka / (i_a - i_sa)
        # 打印完成百分比
        # print(proc / len(dic_seed_k) * 100, '%')

    # print('seed =', s, ' k =', k, ' 竞争性：', comp)

    return comp


def getdic(word):
    # 读取数据
    f = open('140000.TRAIN', 'r', encoding='gbk', errors='ignore')
    data = f.read()
    rows = data.split('\n')

    # word相关词词典
    dic_word = set()

    # word
    for row in rows:
        if row.find(word) != -1:
            segs = jieba.posseg.cut(row)
            for seg in segs:
                dic_word.add(seg.word)

    dic_word.remove(word)

    return dic_word


def intersects(s_dic, k):
    # 读取数据
    f = open('140000.TRAIN', 'r', encoding='gbk', errors='ignore')
    data = f.read()
    rows = data.split('\n')

    '''
        # seed相关词词典
    num_seed = 0

    dic_seed = set()

    # seed
    for row in rows:
        if row.find(s) != -1:
            segs = jieba.posseg.cut(row)
            num_seed = num_seed + 1
            for seg in segs:
                dic_seed.add(seg.word)

    dic_seed.remove(s)
    '''

    dic_seed = s_dic

    # k相关词词典
    num_k = 0

    dic_k = set()

    # k
    for row in rows:
        if row.find(k) != -1:
            segs = jieba.posseg.cut(row)
            num_k = num_k + 1
            for seg in segs:
                dic_k.add(seg.word)

    dic_k.remove(k)
    # print('k =', k, ' 相关词词典：', dic_k)
    # print('k =', k, ' 相关词词典长度：', len(dic_k))

    dic_seed_k = dic_seed & dic_k

    return len(dic_seed_k)


'''
    方法测试
'''
# comp('奔驰', '奥迪')

seed = '奥迪'

read = open('140000_words.TRAIN', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

match_set = set()

match_dic = {}

i = 0

s_dic = getdic(seed)

# print(intersects(s_dic, '宝马'))

k = 1

for word in rows:
    i = i + 1
    if i % 50 != 0:
        continue
    if len(word) == 1:
        continue
    if intersects(s_dic, word) < 20:
        print('%.2f' % (i / 78444 * 100), '%')
        # print('%.2f' % (i / 57130 * 100), '%')
        continue
    # if k == 1:
    #     word = '宝马'
    # if k == 2:
    #     word = '卡宴'
    # if k == 3:
    #     word = '雪佛兰'
    # k = k + 1
    if operator.eq(word, ''):
        print('%.2f' % (i / 78444 * 100), '%')
        # print('%.2f' % (i / 57130 * 100), '%')
        continue
    if operator.eq(word, seed):
        print(i / 78444 * 100, '%')
        continue
    # print(word)
    score = comp_simple(s_dic, seed, word)
    if len(match_dic) < 20:
        match_dic[word] = score
    else:
        for key in list(match_dic.keys()):
            if score > match_dic[key]:
                del match_dic[key]
                match_dic[word] = score
                break
    print('%.2f' % (i / 78444 * 100), '%')
    # print('%.2f' % (i / 57130 * 100), '%')
    # print(len(match_dic))

print('----------final----------')

sort = sorted(match_dic.items(), key=lambda d: d[1], reverse=True)

print(sort)

for item in sort:
    print(item)

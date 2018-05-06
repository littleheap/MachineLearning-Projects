import operator
import jieba.posseg

'''
    #######################################
    1.计算s和k竞争值，基于14W搜索词条，输出过程
    #######################################
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
        # print(i_a)
        # 计算 SA
        i_sa = 0
        for row in rows:
            if row.find(a) != -1:
                if row.find(s) != -1:
                    i_sa = i_sa + 1
        # print(i_sa)
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
            continue
        comp = comp + wa * i_ka / (i_a - i_sa)
        # 打印完成百分比
        # print(proc / len(dic_seed_k) * 100, '%')

    print('seed =', s, ' k =', k, ' 竞争性：', comp)

    return comp


'''
    #########################################
    2.计算s和k竞争值，基于14W搜索词条，不输出过程
    #########################################
'''


def comp_simple(s_dic, s, k):
    # 读取数据
    f = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

    data = f.read()

    rows = data.split('\n')

    # seed相关词词典
    num_seed = 0

    dic_seed = set()

    '''
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

    dic_seed_k = dic_seed & dic_k

    if len(dic_seed_k) == 0:
        return 0

    # 竞争性计算

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
            continue
        comp = comp + wa * i_ka / (i_a - i_sa)

    return comp


'''
    ######################
    3.获得一个词的相关词词典
    ######################
'''


def getdic(word):
    # 读取数据
    f = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

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


'''
    ##########################################################
    4.计算两个词的词典交集长度大小，输入：一个词相关词词典，另一个词
    ##########################################################
'''


def intersects_length(s_dic, k):
    # 读取数据
    f = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

    data = f.read()

    rows = data.split('\n')

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

    dic_seed_k = dic_seed & dic_k

    return len(dic_seed_k)

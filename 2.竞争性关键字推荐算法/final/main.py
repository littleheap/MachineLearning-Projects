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
    read = open('./data/NounWord.1000.Change.TRAIN', 'r', encoding='gbk', errors='ignore')

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
        if operator.eq(word, seed):
            continue
        if funs.intersects_length(s_dic, word) < 10:
            # print('%.2f' % (i / 1200 * 100), '%')
            continue
        score = funs.comp_simple(s_dic, seed, word)
        # print(word)
        # print(score)
        if len(match_dic) < 20:
            match_dic[word] = score
        else:
            sensor = 20
            # 将竞争字典降序排列
            match_dic = dict(sorted(match_dic.items(), key=lambda d: d[1], reverse=True))
            # 找出当前词位于字典中合适位置
            for key in list(match_dic.keys()):
                if match_dic[key] > score:
                    sensor = sensor - 1
                    continue
                if match_dic[key] < score:
                    # 将竞争字典升序排列
                    match_dic = dict(sorted(match_dic.items(), key=lambda d: d[1], reverse=False))
                    # 删除最小的竞争词
                    for min in match_dic:
                        del match_dic[min]
                        break
                    # 插入最大的竞争词
                    match_dic[word] = score
                    break
        # match_dic = dict(sorted(match_dic.items(), key=lambda d: d[1], reverse=True))
        # print(match_dic)
        print('%.2f' % (i / 1200 * 100), '%')

    read.close()

    print('----------final----------')

    # 排序竞争词
    sort = sorted(match_dic.items(), key=lambda d: d[1], reverse=True)

    print(sort)

    for item in sort:
        print(item)


def runs(seed):
    # 设定种子词
    # seed = '奔驰'

    # 读取基础词汇数据
    read = open('./data/NounWord.1000.Handle.TRAIN', 'r', encoding='gbk', errors='ignore')

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
        if operator.eq(word, seed):
            continue
        if funs.intersects_length(s_dic, word) < 10:
            # print('%.2f' % (i / 1200 * 100), '%')
            continue
        score = funs.comp_simple(s_dic, seed, word)
        # print(word)
        # print(score)
        if len(match_dic) < 20:
            match_dic[word] = score
        else:
            sensor = 20
            # 将竞争字典升序排列
            match_dic = dict(sorted(match_dic.items(), key=lambda d: d[1], reverse=False))
            # 找出当前词位于字典中合适位置
            for key in list(match_dic.keys()):
                if match_dic[key] < score:
                    sensor = sensor - 1
                    continue
                if match_dic[key] > score:
                    # 删除最小的竞争词
                    for min in match_dic:
                        del match_dic[min]
                        break
                    # 插入最大的竞争词
                    match_dic[word] = score
                    break
            if sensor == 0:
                # 删除最小的竞争词
                for min in match_dic:
                    del match_dic[min]
                    break
                # 插入最大的竞争词
                match_dic[word] = score

        print('%.2f' % (i / 1200 * 100), '%')

    read.close()

    print('----------final----------')

    # 排序竞争词
    sort = sorted(match_dic.items(), key=lambda d: d[1], reverse=True)

    print(sort)

    for item in sort:
        print(item)


# run('华为')
'''
    ('手机', 0.4730348532778669)
    ('荣耀', 0.11303026317304025)
    ('小米', 0.038434736926811416)
    ('红包', 0.03483528054766144)
    ('三星', 0.02948757406323265)
    ('百度', 0.01727629751148444)
    ('魅族', 0.016404753556152812)
    ('红米', 0.01488174941212883)
    ('北京', 0.008388587467771703)
    ('上海', 0.006540796497391121)
    ('生产', 0.005771758127003496)
    ('大众', 0.005708191858651291)
    ('女生', 0.005452345017797374)
    ('成都', 0.005371904389423415)
    ('电器', 0.004922047193206799)
    ('阿里', 0.004191944359553782)
    ('广州', 0.0038034570674725676)
    ('深圳', 0.00350798841505675)
    ('宝马', 0.0031849318029708662)
    ('条件', 0.0028974912191308466)
'''

# run('北京')
'''
    ('手机', 0.058019524402164874)
    ('上海', 0.04191126941855767)
    ('深圳', 0.040894336528819075)
    ('成都', 0.03286849142992991)
    ('天津', 0.03100646860625945)
    ('广州', 0.02990380070943474)
    ('武汉', 0.02258564539515782)
    ('荣耀', 0.011380806635812169)
    ('百度', 0.011248240922139044)
    ('北京西', 0.010893015108964638)
    ('多长', 0.010096937045687832)
    ('地质', 0.007726440242490968)
    ('女生', 0.0065030777673960585)
    ('小米', 0.00554962745910221)
    ('华为', 0.004709094590398041)
    ('条件', 0.004366412296643915)
    ('男生', 0.0042298731039073735)
    ('朝阳区', 0.003889755586307766)
    ('军训', 0.003727320863196552)
    ('破解版', 0.003370587945143779)
'''

# run('奔驰')
'''
    ('手机', 0.045963883535955416)
    ('奥迪', 0.01998346724742773)
    ('宝马', 0.017273751912385945)
    ('保时捷', 0.009561958150550633)
    ('华为', 0.006561121589284338)
    ('荣耀', 0.006544920961756817)
    ('大众', 0.006462733103140193)
    ('北京', 0.0056672319958332345)
    ('小米', 0.004469623833959575)
    ('三星', 0.0040327401160197)
    ('百度', 0.0038824283980646145)
    ('魅族', 0.0029763245983641393)
    ('男生', 0.0024336723771752134)
    ('上海', 0.002411827108942552)
    ('女生', 0.0021712659222051334)
    ('武汉', 0.0021687741298239)
    ('广州', 0.0017740675291364055)
    ('成都', 0.00164963299276391)
    ('深圳', 0.0013165592317391371)
    ('丰田', 0.0012127236135596998)
'''

# print(funs.getdic('华为'))

run('北京')


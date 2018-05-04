import operator

import jieba.posseg

f = open('140000.TRAIN', 'r', encoding='gbk', errors='ignore')

data = f.read()

rows = data.split('\n')

# 开始数据处理

'''
    ##################
    奔驰&奥迪 竞争性计算
    ##################
'''
print('---------奔驰 奥迪----------')

# 奔驰相关词词典
i_benz = 0

dic_benz = set()

# 奔驰
for row in rows:
    if row.find('奔驰') != -1:
        seg = jieba.posseg.cut(row)
        i_benz = i_benz + 1
        for s in seg:
            dic_benz.add(s.word)

dic_benz.remove('奔驰')
print('奔驰相关词词典：', dic_benz)
print('奔驰相关词词典长度：', len(dic_benz))  # >>> 129

# 奥迪相关词词典
i_audi = 0

dic_audi = set()

# 奔驰
for row in rows:
    if row.find('奥迪') != -1:
        seg = jieba.posseg.cut(row)
        i_audi = i_audi + 1
        for s in seg:
            dic_audi.add(s.word)

dic_audi.remove('奥迪')
print('奥迪相关词词典：', dic_audi)
print('奥迪相关词词典长度：', len(dic_audi))  # >>> 140

# 奔驰奥迪相关性交集
dic_benz_audi = dic_benz & dic_audi

print('奔驰 奥迪 相关词交集：', dic_benz_audi)
print('奔驰 奥迪 相关词交集长度：', len(dic_benz_audi))  # >>> 39

# 竞争性计算
print('---------竞争性计算----------')

s = '奔驰'
k = '奥迪'
comp = 0

# 计算 S
i_s = 0

for row in rows:
    if row.find(s) != -1:
        i_s = i_s + 1

# 遍历中介词
proc = 0
for a in dic_benz_audi:
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
    print(proc / len(dic_benz_audi) * 100, '%')

print('奔驰奥迪竞争性：', comp)  # >>> 0.019983097403486694



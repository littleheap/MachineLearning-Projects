import jieba
import jieba.posseg
import os

f = open('user_tag_query.10W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = f.read()

rows = data.split('\n')

# print(len(rows))

'''
    种子关键字 s：奔驰
    任意关键字 k：奥迪
    中介词 a：雨刷
'''

i_benz = 0

dic_benz = set()

# 奔驰
for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('奔驰') != -1:
            seg = jieba.posseg.cut(msg)
            i_benz = i_benz + 1
            # print(i, ' : ', end='')
            for s in seg:
                # print(s.word, '|', end='')
                dic_benz.add(s.word)
            # print('\n')

print('奔驰相关词词典：', dic_benz)
print('奔驰相关词词典长度：', len(dic_benz))

# 奥迪
i_audi = 0

dic_audi = set()

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('奥迪') != -1:
            seg = jieba.posseg.cut(msg)
            i_audi = i_audi + 1
            # print(i, ' : ', end='')
            for s in seg:
                # print(s.word, '|', end='')
                dic_audi.add(s.word)
            # print('\n')

print('奥迪相关词词典：', dic_audi)
print('奥迪相关词词典长度：', len(dic_audi))

# 奔驰奥迪相关性交集
dic_benz_audi = dic_benz & dic_audi

print('奔驰 奥迪 相关词交集：', dic_benz_audi)
print('奔驰 奥迪 相关词交集长度：', len(dic_benz_audi))

# 竞争性计算
print('---------奔驰 奥迪----------')

s = '奔驰'
k = '奥迪'
comp = 0

# s = 奔驰
i_s = 0

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('奔驰') != -1:
            i_s = i_s + 1

# 遍历中介词
proc = 0
for a in dic_benz_audi:
    proc = proc + 1
    i_a = 0
    for row in rows:
        split_row = row.split("\t")
        for msg in split_row:
            if msg.find(a) != -1:
                i_a = i_a + 1
    i_sa = 0
    for row in rows:
        split_row = row.split("\t")
        for msg in split_row:
            if msg.find(a) != -1:
                if msg.find(s) != -1:
                    i_sa = i_sa + 1
    i_ka = 0
    for row in rows:
        split_row = row.split("\t")
        for msg in split_row:
            if msg.find(a) != -1:
                if msg.find(k) != -1:
                    i_ka = i_ka + 1
    wa = i_sa / i_s
    comp = comp + wa * i_ka / (i_a - i_sa)
    print(proc / len(dic_benz_audi) * 100, '%')

print('奔驰奥迪竞争性：', comp)
'''
# s = 奔驰
i_benz = 0

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('奔驰') != -1:
            i_benz = i_benz + 1

print('s = 奔驰：', i_benz)

# a = 雨刷
i_yushua = 0

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('雨刷') != -1:
            i_yushua = i_yushua + 1

print('a = 雨刷：', i_yushua)

# a = 雨刷 s = 奔驰
i_yushua_benz = 0

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('雨刷') != -1:
            if msg.find('奔驰') != -1:
                i_yushua_benz = i_yushua_benz + 1

print('a = 雨刷 s = 奔驰：', i_yushua_benz)

# a = 雨刷 k = 奥迪
i_yushua_zudi = 0

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('雨刷') != -1:
            if msg.find('奥迪') != -1:
                i_yushua_zudi = i_yushua_zudi + 1

print('a = 雨刷 k = 奥迪：', i_yushua_zudi)

'''

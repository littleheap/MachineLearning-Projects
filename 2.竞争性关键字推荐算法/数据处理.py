import jieba
import jieba.posseg

f = open('user_tag_query.10W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = f.read()

rows = data.split('\n')

print(len(rows))

i = 0

# 读取带有‘  ’的搜索词条
for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        if msg.find('比亚迪') != -1:
            seg = jieba.posseg.cut(msg)
            i = i + 1
            print(i, ' : ', end='')
            for s in seg:
                print(s.word, '|', end='')
            print('\n')

'''
f = open('user_tag_query.10W.TEST', 'r', encoding='gbk', errors='ignore')

data = f.read()

rows = data.split('\n')

print(len(rows))

for i in range(5):
    split_row = rows[i].split("\t")
    for msg in split_row:
        # print(msg)
        pass
    print('-----------------------------')
'''

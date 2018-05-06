import jieba.posseg

'''
    整理出14W条搜索数据中，全部7W个词组
'''

read = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

alldic = set()

allwords = ''

for row in rows:
    segs = jieba.posseg.cut(row)
    for seg in segs:
        alldic.add(seg.word)

print(len(alldic))  # >>> 77273

read.close()

# 写入文件
write = open("./data/Word.7W.TRAIN", "w")

for word in alldic:
    allwords = allwords + '\n' + word

write.write(allwords)

write.close()

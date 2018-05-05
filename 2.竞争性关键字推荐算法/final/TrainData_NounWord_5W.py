import jieba.posseg

'''
    整理出7W条词组中，全部5W个名词
'''

read = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

alldic = set()

allwords = ''

for row in rows:
    segs = jieba.posseg.cut(row)
    for seg in segs:
        if seg.flag.find('n') != -1:
            alldic.add(seg.word)

print(len(alldic))  # >>> 56164

read.close()

# 写入文件
write = open("./data/NoneWord.5W.TRAIN", "w")

for word in alldic:
    allwords = allwords + '\n' + word

write.write(allwords)

write.close()

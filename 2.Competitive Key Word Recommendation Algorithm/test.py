import operator
import jieba.posseg

# read = open('./data/Origion.10W.TRAIN', 'r', encoding='gbk', errors='ignore')
#
# data = read.read()
#
# rows = data.split('\n')
#
# for row in rows:
#     print(row)

segs = jieba.posseg.cut('宝马和奔驰那个好')

for word in segs:
    print(word.word)

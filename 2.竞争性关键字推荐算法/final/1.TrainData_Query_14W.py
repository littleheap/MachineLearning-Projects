'''
    整理出Train训练集10W条个人搜索数据中，全部1400W条搜索，缩水1/100，得到14W条搜索记录
'''

# 读取文件
import operator

read = open('./data/Origion.10W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

content = ""

number = 0

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        number = number + 1
        if number % 100 == 0:
            if len(msg) == 32:
                continue
            if operator.eq(msg, '0') or operator.eq(msg, '1') or operator.eq(msg, '2') or operator.eq(msg,
                                                                                                      '3') or operator.eq(
                msg, '4') or operator.eq(msg, '5') or operator.eq(msg, '6'):
                continue
            content = content + "\n" + msg

# 写入文件
write = open("./data/Query.14W.TRAIN", "w")

write.write(content)

read.close()

write.close()

# 查看训练集Word大小
read = open('./data/Query.14W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

# 输出整理后的搜索记录条数
print(len(rows))
# >>> 142708

read.close()

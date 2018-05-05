# 读取文件
read = open('user_tag_query.10W.TRAIN', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

# 写入文件
write = open("140000.TRAIN", "w")

# print(len(rows))

number = 0

content = ""

for row in rows:
    split_row = row.split("\t")
    for msg in split_row:
        number = number + 1
        if number % 100 == 0:
            content = content + "\n" + msg

write.write(content)

read.close()

write.close()

# print(number)

# 查看缩水训练集大小
read = open('140000.TEST', 'r', encoding='gbk', errors='ignore')

data = read.read()

rows = data.split('\n')

print(len(rows))
# >>> 146730

read.close()

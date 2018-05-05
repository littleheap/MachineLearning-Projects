# import re
#
# print(len('22DD920316420BE2DF8D6EE651BA174B'))  # >>> 32
#
# word = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", '奥迪')
# print(word)

# print(type(0.0026455339490908817))

dic = {'a': 1, 'b': 2, 'c': 4, 'd': 3}
dic = sorted(dic.items(), key=lambda d: d[1], reverse=False)
print(dic)
print(type(dic))
dic = dict(dic)
print(dic)
print(type(dic))
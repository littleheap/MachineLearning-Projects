'''
dic = {'python', 'Java'}

print(dic)

dic.add('python')

print(dic)

dic.add('c++')

print(dic)

dic1 = {'a', 'b', 'c'}
dic2 = {'a', 'b', 'd'}
print(dic1 & dic2)

for i in dic:
    print(i)

print(1/3)
'''

match = {}

match['hello'] = 1
match['sss'] = 2
match['aaa'] = 4
match['qqq'] = 3

sort = sorted(match.items(), key=lambda d: d[1], reverse=True)
print(sort)

for name in match:
    print(name)
    print(match[name])

print(len(match))

del match['hello']

for name in match:
    print(name)
    print(match[name])

print(len(match))
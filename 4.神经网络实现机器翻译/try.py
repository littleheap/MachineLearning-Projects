ls = {'a': 5, 'b': 10, 'c': 3, 'd': 2, 'e': 1}

for s in enumerate(ls):
    print(s)


for (key, value) in enumerate(ls):
    print((key, value))


word_dict = {w[0]: index + 1 for (index, w) in enumerate(ls)}

print(word_dict)

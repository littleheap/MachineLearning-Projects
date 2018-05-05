import re

print(len('22DD920316420BE2DF8D6EE651BA174B'))  # >>> 32

word = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", '奥迪')
print(word)

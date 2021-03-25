import re
from collections import Counter
catglorydict1=Counter()
iddict1=Counter()
idset1=set()
count=0
# sort by frequency, then alphabetically
idset2 = set()
with open(r"./dataset/data1.csv", 'r') as f:
    for line in f:
        spacelist = line.split(',')
        # print(spacelist)
        lenspacelist = len(spacelist)
        for i in range(lenspacelist - 1):
            cattmp = re.findall('#(.*)@', spacelist[i + 1])[0]
            idtmp = re.findall('(.*)#', spacelist[i + 1])[0]
            if idtmp not in idset2:
                catglorydict1[cattmp] -= 1
                iddict1[idtmp] -= 1
                idset2.add(idtmp)
words_and_frequencies = sorted(catglorydict1.items(), key=lambda tup: tup[0])
words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
with open(r"./output/catgloryCount5.txt",'w') as f:
    for word, freq in words_and_frequencies:
        f.write(str(word)+","+str(freq)+'\n')
print(count)

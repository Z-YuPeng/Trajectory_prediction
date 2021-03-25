import re
import numpy as np
import time
from tqdm import tqdm
from dataoperator import WordVocab
vocab = WordVocab.load_vocab("./output/markovvocab")
texts = []
dict = {}
with open("dataset/data1.csv") as f:
    for line in f:
        spacelist = line.split(',')[1:]
        lenspacelist = len(spacelist)
        sentence = []
        idsentence=[]
        for i in range(lenspacelist):
            cattmp = re.findall('#(.*)@', spacelist[i])[0]
            idtmp = re.findall('(.*)#', spacelist[i])[0]
            idsentence.append(idtmp)
            if cattmp == "Null":
                sentence.append(-1)
            else:
                sentence.append(vocab.stoi[cattmp])
        for i in range(len(sentence)):
            idtmp=idsentence[i]
            if i > 0 and i < lenspacelist - 1:
                if sentence[i - 1] != -1 and sentence[i + 1] != -1:
                    if idtmp in dict:
                        dict[idtmp].append([sentence[i - 1], sentence[i + 1]])
                    else:
                        dict[idtmp]=[[sentence[i - 1], sentence[i + 1]]]
        texts.append(sentence)
probality = np.ones((103, 103, 103),dtype=np.float)
for sentence in tqdm(texts):
    lensentence = len(sentence)
    for i in range(1,lensentence-1):
        if sentence[i] != -1 and sentence[i-1] != -1 and sentence[i+1] != -1:
            probality[sentence[i-1]][sentence[i+1]][sentence[i]] += 1
for i in range(103):
    for j in  range(103):
        sum=0
        for k in range(103):
            sum += probality[i][j][k]
        for k in range(103):
            probality[i][j][k] /= sum
answerdict={}
for idtmp in dict.keys():
    listtmp=dict[idtmp]
    protmp= probality[listtmp[0][0]][listtmp[0][1]]
    for i in range(len(listtmp)-1):
        protmp *= probality[listtmp[i+1][0]][listtmp[i+1][1]]
    max = 0
    maxid = 0
    for i in range(103):
        if protmp[i]>max:
            max=protmp[i]
            maxid=i
    answerdict[idtmp]=vocab.itos[maxid]
testoutline=[]
with open(r"./dataset/data3.csv",'r') as f:
    for line in f:
        line = line.replace('\n','')
        spacelist=line.split(',')
        testoutline.append(spacelist[0])

with open(r"./output/predict13.csv",'w') as f:
    for i in range(len(testoutline)):
        if testoutline[i] not  in answerdict.keys():
            answerdict[testoutline[i]]="Null"
        f.write(testoutline[i]+','+answerdict[testoutline[i]]+'\n')
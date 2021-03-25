import torch
from dataoperator import  WordVocab
def sentenceToId(sentence,vocab):
    for i, token in enumerate(sentence):
        if token != 'Null':
            sentence[i] = vocab.stoi.get(token, vocab.unk_index)
        else:
            sentence[i] = vocab.mask_index
    if len(sentence)<32:
        for j in  range(len(sentence),32):
            sentence.append(vocab.pad_index)

    return sentence
import re

def softmax(x):
    softmaxer = torch.nn.LogSoftmax(dim=-1)
    x= torch.tensor(x)
    x= softmaxer(x)
    return x.numpy()
import numpy as np
from tqdm import tqdm
if __name__ == "__main__":
    compensation = np.ones((108), dtype=np.float)
    for i in range(10):
        compensation[i]=10000
    device='cuda'
    print("Loading Vocab")
    vocab = WordVocab.load_vocab("./output/vocab")
    print("Vocab Size: ", len(vocab))

    print("Building BERT model")
    model = torch.load('./output/model.ep9')
    model.to(device)
    model.eval()
    dict={}
    inputdict={}
    count=0
    with open(r"./dataset/data1.csv", 'r') as f:
        for line in tqdm(f):
            spacelist = line.split(',')
            lenspacelist = len(spacelist)
            text=[]
            tmpidlist=[]
            nullindex=[]
            for i in range(lenspacelist - 1):
                cattmp = re.findall('#(.*)@', spacelist[i + 1])[0]
                idtmp = re.findall('(.*)#', spacelist[i + 1])[0]
                text.append(cattmp)
                if cattmp == 'Null':
                    tmpidlist.append(idtmp)
                    nullindex.append(i)
            for i,index in enumerate(nullindex):
                if tmpidlist[i] not in inputdict.keys():
                    inputdict[tmpidlist[i]]=[]
                texttmp=[cattmp for i,cattmp in enumerate(text) if i==index or cattmp!='Null']
                for k,cat in enumerate(texttmp):
                    if cat=='Null':
                        index=k
                        break
                if len(texttmp)<32:
                    inputdict[tmpidlist[i]].append(sentenceToId(texttmp,vocab))
                elif index<20:
                    inputdict[tmpidlist[i]].append(sentenceToId(texttmp[0:32], vocab))
                elif index +12 >len(text):
                    inputdict[tmpidlist[i]].append(sentenceToId(texttmp[len(texttmp)-32:len(texttmp)], vocab))
                else:
                    inputdict[tmpidlist[i]].append(sentenceToId(texttmp[index-20:index+11], vocab))



    for idtmp in tqdm(inputdict.keys()):
        probability = np.zeros((108), dtype=np.float)
        xer=inputdict[idtmp]
        inputbatch=[]
        nullarray=[]
        for inputtext in xer:
            nullindexer=0
            for i,cat in enumerate(inputtext):
                if cat == vocab.mask_index:
                    nullindexer = i
                    break
            nullarray.append(nullindexer)
            inputbatch.append(inputtext)
        if len(inputbatch)<=300:
            inputbatch = torch.tensor(inputbatch)
            inputbatch = inputbatch.to(device)
            output = (model(inputbatch)).detach().cpu().numpy()
            for i, sentence in enumerate(output.tolist()):
                probability += sentence[nullarray[i]]
        else:
            for minbatch in range(300,len(inputbatch),300):
                nowbatch= inputbatch[minbatch-300:minbatch]
                nownullarray=nullarray[minbatch-300:minbatch]
                nowbatch = torch.tensor(nowbatch)
                nowbatch = nowbatch.to(device)
                output = (model(nowbatch)).detach().cpu().numpy()
                for i, sentence in enumerate(output.tolist()):
                    probability += sentence[nownullarray[i]]
        probability*=compensation
        cat = np.argmax(probability)
        dict[idtmp]=vocab.itos[cat]
    testoutline = []
    outputY=[]
    with open(r"./dataset/data3.csv", 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            spacelist = line.split(',')
            testoutline.append(spacelist[0])
            if spacelist[0] not in dict:
                outputY.append("Null")
            else :
                outputY.append(dict[spacelist[0]])
    with open(r"./output/predict14.csv", 'w') as f:
        for i in range(len(testoutline)):
            f.write(testoutline[i] + ',' + outputY[i] + '\n')
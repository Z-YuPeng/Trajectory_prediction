import torch
import argparse
from torch.utils.data import DataLoader
from ourDeepLearingModel.bert import BERT
from DeeplearningTrainer import BERTTrainer
from dataoperator import BERTDataset, WordVocab
def sentenceToId( sentence,vocab):
    for i, token in enumerate(sentence):
        if token !='Null':
            sentence[i] = vocab.stoi.get(token, vocab.unk_index)
        else:
            sentence[i]=vocab.mask_index
    return torch.tensor([sentence])
import re
from tqdm import tqdm
if __name__ == "__main__":
    device='cuda'
    print("Loading Vocab")
    vocab = WordVocab.load_vocab("./output/vocab")
    print("Vocab Size: ", len(vocab))
    print("Building BERT model")
    model = torch.load('./output/model.ep9')
    model.to(device)
    model.eval()
    dict={}
    count=0
    with open(r"./dataset/data1.csv", 'r') as f:
        for line in tqdm(f):
            spacelist = line.split(',')
            # print(spacelist)
            lenspacelist = len(spacelist)
            text=[]
            tmpidlist=[]
            nullindex=[]
            for i in range(lenspacelist - 1):
                cattmp = re.findall('#(.*)@', spacelist[i + 1])[0]
                idtmp = re.findall('(.*)#', spacelist[i + 1])[0]
                text.append(cattmp)
                if cattmp=='Null':
                    tmpidlist.append(idtmp)
                    nullindex.append(i)
            inputtext= sentenceToId(text,vocab)
            inputtext=inputtext.to(device)
            if len(inputtext[0])> 512:
                count+=1
                continue
            output = torch.argmax(model(inputtext),dim=2).squeeze(0).cpu().numpy()
            for i in range(len(nullindex)):
                dict[tmpidlist[i]]=vocab.itos[output[nullindex[i]]]
        testoutline = []
        outputY=[]
        print(count)
        with open(r"./dataset/data3.csv", 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                spacelist = line.split(',')
                testoutline.append(spacelist[0])
                if spacelist[0] not in dict:
                    outputY.append("Null")
                else :
                    outputY.append(dict[spacelist[0]])
        with open(r"./output/predict8.csv", 'w') as f:
            for i in range(len(testoutline)):
                f.write(testoutline[i] + ',' + outputY[i] + '\n')







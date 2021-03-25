import csv
import gensim
from gensim.models import word2vec
import numpy as np
import scipy.io as sio

loc_catglory = dict()
loc_test = dict()

with open('data1.csv','r') as f:
    with open('datatest.txt', 'w') as ff:
        f_csv = csv.reader(f)
        for row in f_csv:
            for i in range(1,len(row)):
                loc1 = row[i].find("#")
                loc2 = row[i].find("@")

                locId = row[i][:loc1]
                catglory = row[i][loc1+1:loc2]
                if(catglory!="Null"):
                    loc_catglory[locId] = catglory
                else:
                    loc_test[locId] = catglory
                # row[i] = row[i][:loc1]+"_"+row[i][loc2+1:]
                row[i] = row[i][:loc1]
                ff.write(row[i]+" ")
            ff.write("\n")

test_labels = dict()
with open('data3.csv','r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        test_labels[row[0]] = row[1]



vec_size = 300

sentences=word2vec.Text8Corpus("datatest.txt")

model=gensim.models.Word2Vec(sentences,sg=1,size=vec_size,window=10,min_count=1,negative=5,sample=0.001,hs=1,workers=4)
model.save("ML_EXP")	#模型会保存到该 .py文件同级目录下，该模型打开为乱码
# model.wv.save_word2vec_format("AL_EXP",binary = "False")  #通过该方式保存的模型，能通过文本格式打开，也能通过设置binary是否保存为二进制文件。但该模型在保存时丢弃了树的保存形式（详情参加word2vec构建过程，以类似哈夫曼树的形式保存词），所以在后续不能对模型进行追加训练

#该步骤也可分解为以下三步（但没必要）：
#model=gensim.model.Word2Vec() 建立一个空的模型对象
#model.build_vocab(sentences) 遍历一次语料库建立词典
#model.train(sentences) 第二次遍历语料库建立神经网络模型

#sg=1是skip—gram算法，对低频词敏感，默认sg=0为CBOW算法
#size是神经网络层数，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
#window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）
#min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
#negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3,
#negative: 如果>0,则会采用negativesamping，用于设置多少个noise words
#hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
#workers是线程数，此参数只有在安装了Cpython后才有效，否则只能使用单核
# min_count，是去除小于min_count的单词
# size，神经网络层数
# sg， 算法选择
# window， 句子中当前词与目标词之间的最大距离
# workers，线程数


#对..wv.save_word2vec_format保存的模型的加载：
# model = model.wv.load_word2vec_format('模型文件名')


labels = set()
for key in loc_catglory:
    labels.add(loc_catglory[key])
labels = list(labels)

model = gensim.models.Word2Vec.load("ML_EXP")
train_X = np.zeros((len(loc_catglory),vec_size))
train_Y = np.zeros((len(loc_catglory),1))
test_X = np.zeros((len(loc_test),vec_size))

for index, key in enumerate(loc_catglory):
    train_X[index,:] = model[str(key)]
    train_Y[index,0] = labels.index(loc_catglory[key])
for index, key in enumerate(loc_test):
    test_X[index,:] = model[str(key)]

sio.savemat('train_data.mat',{'train_x':train_X})
sio.savemat('train_labels.mat',{'train_y':train_Y})
sio.savemat('test_data.mat',{'test_x':test_X})

with open('labels-ids.txt', 'w') as ff:
    for i in range(len(labels)):
        ff.write(labels[i]+ "_"+ str(i))
        ff.write("\n")


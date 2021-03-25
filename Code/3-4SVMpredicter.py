import re
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
dict = {}
corpus = []
X = []
Y = []
datedict = {}
import time
with open(r"./dataset/data1.csv",'r') as f:
    for line in f:
        spacelist=line.split(',')
        # print(spacelist)
        lenspacelist = len(spacelist)
        sentence=[]
        for i in range(lenspacelist-1):
            cattmp=re.findall('#(.*)@',spacelist[i+1])[0]
            idtmp=re.findall('(.*)#',spacelist[i+1])[0]
            sentence.append(idtmp)
            if  idtmp in dict :
                if dict[idtmp]=='null':
                    dict[idtmp]=cattmp
                else:
                    if dict[idtmp]!=cattmp:
                        print("重复且不相等")
                        print(dict[idtmp],cattmp,idtmp)
            else:
                dict[idtmp] = cattmp
        corpus.append(sentence)
idlist=list(dict.keys())
print(idlist)
print(len(idlist))
model = Word2Vec(corpus, sg=1, size=150,  window=10,  min_count=1,  negative=10, sample=0.001, hs=1, workers=4)
print(type(model.wv.vocab))
for i in range(len(idlist)):
    if dict[idlist[i]]=='Null':
        continue
    X.append(list(model.wv[idlist[i]]))
    Y.append(dict[idlist[i]])
# with open(r"./output/word2vecfile.txt",'w') as f:
#     for i in range(len(idlist)):
#         f.write(idlist[i]+':'+str(X[i])+"\n")
le = preprocessing.LabelEncoder()
Ynew = le.fit_transform(Y)
# knnclassifier = KNeighborsClassifier(n_neighbors=5,  weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# knnclassifier.fit(X,Ynew)
# print(knnclassifier.predict([X[1]]) == [Ynew[1]])
testoutline=[]
testX=[]
min_max_scaler = preprocessing.StandardScaler()
X = min_max_scaler.fit_transform(X)
 # rbf核函数，设置数据权重
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
svc = SVC(kernel='rbf', class_weight='balanced')
c_range = np.logspace(-3, 9, 100, base=2)
gamma_range = np.logspace(-9, 6, 20, base=2)
# 网格搜索交叉验证的参数范围，cv=3,3折交叉
param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
# 训练模型
clf = grid.fit(X, Ynew)
print(grid.best_params_)
# # 计算测试集精度
# score = grid.score(x_test, y_test)
# print('精度为%s' % score)
# MLPclassifier.fit(X,Ynew)
with open(r"./dataset/data3.csv",'r') as f:
    for line in f:
        line = line.replace('\n','')
        spacelist=line.split(',')
        testoutline.append(spacelist[0])
        testX.append(model.wv[spacelist[0]])
# testY = knnclassifier.predict(testX)
testX = min_max_scaler.transform(testX)
testY = clf.predict(testX)
outputY = le.inverse_transform(testY)
with open(r"./output/predict20.csv",'w') as f:
    for i in range(len(testoutline)):
        f.write(testoutline[i]+','+outputY[i]+'\n')
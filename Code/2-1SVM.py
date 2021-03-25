import re
from gensim.models import Word2Vec
from sklearn import preprocessing
dict = {}
corpus = []
X = []
Y = []
datedict = {}
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
model = Word2Vec(corpus, sg=1, size=150,  window=10,  min_count=1,  negative=5, sample=0.001, hs=1, workers=4)
for i in range(len(idlist)):
    if dict[idlist[i]]=='Null':
        continue
    X.append(list(model.wv[idlist[i]]))
    Y.append(dict[idlist[i]])
le = preprocessing.LabelEncoder()
Ynew = le.fit_transform(Y)
testoutline=[]
testX=[]
min_max_scaler = preprocessing.StandardScaler()
X = min_max_scaler.fit_transform(X)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
svc = SVC(kernel='rbf', class_weight='balanced',C=42.22425,gamma=0.0078125)
# 训练模型
# a=cross_val_score(svc,X, Ynew,cv=5,scoring='f1_micro')
# count = sum(a)
# print(a)
# print(count/5)
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
testY =svc.predict(testX)
outputY = le.inverse_transform(testY)
with open(r"./output/predict1.csv",'w') as f:
    for i in range(len(testoutline)):
        f.write(testoutline[i]+','+outputY[i]+'\n')

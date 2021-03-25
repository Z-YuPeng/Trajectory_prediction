from gensim.models import Word2Vec
import numpy as np

import csv
#读取csv文件
with open("./data1.csv", "r") as f:
    reader = csv.reader(f)
    data1list=list(reader)
    for row in data1list:
        row.pop(0)#除去第一列
# print(len(data1list))
rownum=len(data1list)
#将数据进行处理
# 已知分类位置保存其分类，未知分类未知保存其未知id
for i in range(rownum):
    for j in range(len(data1list[i])):
        temp=data1list[i][j]
        id=temp[0:24]
        cl = temp[25:temp.find('@')]
        if cl=='Null':
            data1list[i][j]=id+'zzz'
        else:
            data1list[i][j]=cl
# print(data1list)
#
model = Word2Vec(data1list, sg=0, size=100,  window=10,  min_count=1,  negative=5, sample=0.001, hs=1, workers=4)

#保存和加载模型
# model.save("./model2")
# model=gensim.models.Word2Vec.load("model2")

#加载所有未知分类的id列表
with open("./data3.csv", "r") as f:
    reader = csv.reader(f)
    data3list=list(reader)
for i in range(len(data3list)):
    data3list[i]=data3list[i][0]+'zzz'
print(data3list)
num=len(data3list)

#所有的类别
with open("./catglory.txt","r") as f:
    reader=csv.reader(f)
    cate=list(reader)
catlist=[]
for i in range(len(cate)):
    catlist.append(cate[i][0])
print(catlist)

#计算idd向量与每一个分类向量之间的距离
predlist=[]
right=0
for k in range(len(data3list)):
    vec1=model[data3list[k]]
    dist_min=100000000000
    cl=''
    for j in range(len(catlist)):
        temp=catlist[j]
        vec2 = model[temp]
        dist = np.linalg.norm(vec1 - vec2)
        # if temp=='Train Station':
        #     dist=dist*1.001
        if dist<dist_min:
            dist_min=dist
            cl=temp
    tlist=[data3list[k][:24],cl]
    predlist.append(tlist)

print(predlist)

# 1. 创建文件对象
f = open('predict2.csv','w',encoding='utf-8',newline='')
# 2. 基于文件对象构建 csv写入对象
csv_writer = csv.writer(f)
for m in range(len(predlist)):
    csv_writer.writerow(predlist[m])

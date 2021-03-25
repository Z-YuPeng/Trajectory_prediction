import matplotlib.pyplot as plt
import matplotlib
x=[]
labels=[]
count=0
matplotlib.rc("font",family='MicroSoft YaHei',weight="bold")
with open(r"./output/catgloryCount.txt",'r') as f:
    for i,line in enumerate(f):
        if i <6:
            line = line.replace('\n','')
            spacelist=line.split(',')
            x.append(int(spacelist[1]))
            labels.append(spacelist[0])
        else:
            line = line.replace('\n', '')
            spacelist = line.split(',')
            count+=int(spacelist[1])
labels.append("其他")
x.append(count)
patches,l_text,p_text =plt.pie(x,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
for t in l_text:
    t.set_size(15)
for t in p_text:
    t.set_size(15)
plt.show()

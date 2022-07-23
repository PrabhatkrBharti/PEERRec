import pandas as pd
import os
import json
from summa.summarizer import summarize

year="2017"
file = os.listdir("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/paper")
file.sort()
names = []

newdf = pd.DataFrame(columns = ['paperid','sectionid','New_Summary'])
# file =  pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/2018/files_names.csv")
count=0
txt = []
papername = []
id = []
zerocount = 0
cpid=0
for i in range(len(file)):
    f = file[i]
    df = pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/Paper/"+f)
    # df = pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/2018/2018/Papers_CSV/"+f+'.paper.jsoncsv.csv')
    df.dropna(subset=['text'], inplace=True)
    joined_string=''
    flag=False
    for row, col in df.iterrows():
        txt1 = col['text']
        summary = summarize(txt1, words = 72) #72 - 28, 100 - 28, 130 
        
        if len(summary) == 0:
            zerocount +=1
            continue
        
        # print(summary)
        # print(len(summary))
        # print('-------------------')
        # joined_string+=col['text']
    # summary = summarize(joined_string, words = 150)
        
        txt.append(summary)
        papername.append(cpid)
        id.append(count)
        count += 1
        flag=True
    # print(summary)
    # print(len(summary))
    # print("-------------------------------")
    if flag==True:
        cpid+=1
        names.append([f[0:len(f)-15]])
               
print(zerocount)
# print(joined_string)
newdf['paperid'] = papername
newdf['sectionid'] = id
newdf['New_Summary'] = txt
 
newdf.to_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/Paper_Section_Summary.csv", index=False)
names = pd.DataFrame(names, columns = ['names'])
names.to_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/final_files_names.csv", index=False)
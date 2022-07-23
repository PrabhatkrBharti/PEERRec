# -*- coding: utf-8 -*-


import pandas as pd
import os
import json
from summa.summarizer import summarize

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--papers_pdf', type=str, default='./paper')
args = parser.parse_args()

data_path = args.papers_pdf+'/'

file = os.listdir(data_path)
file.sort()
names = []

newdf = pd.DataFrame(columns = ['paperid','sectionid','New_Summary'])
count=0
txt = []
papername = []
id = []
zerocount = 0
cpid=0
for i in range(len(file)):
    f = file[i]
    df = pd.read_csv("/content/drive/My Drive/Paper/"+f)
    df.dropna(subset=['text'], inplace=True)
    joined_string=''
    flag=False
    for row, col in df.iterrows():
        txt1 = col['text']
        summary = summarize(txt1, words = 72) #72 - 28, 100 - 28, 130 
        
        if len(summary) == 0:
            zerocount +=1
            continue
        

        
        txt.append(summary)
        papername.append(cpid)
        id.append(count)
        count += 1
        flag=True

    if flag==True:
        cpid+=1
        names.append([f[0:len(f)-15]])
               
print(zerocount)

newdf['paperid'] = papername
newdf['sectionid'] = id
newdf['New_Summary'] = txt
 
newdf.to_csv(data_path+"Paper_Section_Summary.csv", index=False)
names = pd.DataFrame(names, columns = ['names'])
names.to_csv(data_path+"final_files_names.csv", index=False)


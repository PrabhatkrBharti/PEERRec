import pandas as pd
import os
import json


ext = ('.json')
year = "2020"
papername = []
file =  pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/final_files_names.csv")
paper = []

paper1=[]
for files in file:
    if not files.__contains__('.paper'):
        # print(files)
        paper.append(files)
    if files.__contains__('.paper'):
        paper1.append(files)        
        
# paper.remove('csv')
# paper.remove('reviewcsv')
i=0
paper=paper[:30]
paper1=paper1[:30]
for i in range(len(paper)):
    df = pd.read_json("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/"+year+"/"+paper[i])
    abst = df['abstract']
    sections = df['reviews']
    score = df['score']
    verdict = df['verdict']
    year = df['year']     
    comments = []
    recommendation =[]
    title= []
    for section in sections[:3]:
        recommendation.append(section['recommendation'])
        comments.append(section['comments'])
        title.append(section['title'])
                                
    newdf = pd.DataFrame(columns = ['abs','recommendation','comments','title'])                        
    newdf['abs'] = str(abst)
    newdf['recommendation']=recommendation
    newdf['comments']=comments
    newdf['title']=title
    newdf['score']=score
    newdf['verdict']=verdict
    newdf['year']=year
    full_text = ''
    df = pd.read_json("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/"+year+"/"+paper1[i])
    for j in df['metadata']['sections']:
        full_text += j['text']
                            # print(full_text)
    newdf['paper_text']=full_text
    
    # csv_name = "file_"+str(i)+".csv"
    # newdf.to_csv(csv_name)
    newdf.to_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/Reviews/"+str(i)+'csv.csv')
    
                
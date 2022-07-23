import pandas as pd
import os
import json

import pandas as pd
import os
import json

newdf = pd.DataFrame(columns = ['id','r1','r2','r3','label','rec1','rec2','rec3'])
year = "2017"
papername = []

file =  pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/final_files_names.csv")
count=0
for i in range(len(file)):
    if i%100==0:
        print(i,'/',len(file))
    f = str(file['names'][i])
    df = pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/Review/"+f+".jsoncsv.csv")

    r1 = df['comments'][0]
    r2 = df['comments'][1]
    r3 = df['comments'][2]
    label = df['verdict'][0]
    rec1 = df['recommendation'][0]
    rec2 = df['recommendation'][1]
    rec3 = df['recommendation'][2]
    id = f
    newdf.loc[count]=[count,r1,r2,r3,label,rec1,rec2,rec3]
    count+=1
    
newdf.to_csv("C:/Users/Viswash/Desktop/IIT Research/Data/"+year+"/Final_Reviews_ALL1.csv", index=False)



# newdf = pd.DataFrame(columns = ['id','r1','r2','r3','label','rec1','rec2','rec3'])

# papername = []

# file =  pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/2018/files_names.csv")
# count=0
# for i in range(804):
#     f = file['names'][i]
#     df = pd.read_csv("C:/Users/Viswash/Desktop/IIT Research/Data/2018/2018/reviewcsv/"+f+'.jsoncsv.csv')

#     r1 = df['comments'][0]
#     r2 = df['comments'][1]
#     r3 = df['comments'][2]
#     label = df['verdict'][0]
#     rec1 = df['recommendation'][0]
#     rec2 = df['recommendation'][1]
#     rec3 = df['recommendation'][2]
#     id = f
#     # papername.append(f)
#     newdf.loc[count]=[count,r1,r2,r3,label,rec1,rec2,rec3]
#     # print(newdf)
#     count+=1
    
# newdf.to_csv("Final_Reviews_ALL.csv", index=False)

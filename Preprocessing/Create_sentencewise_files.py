# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./2017 Data')
args = parser.parse_args()

data_path = args.dataset+'/'

df = pd.read_csv(data_path + "Data.csv", encoding='utf8')
df.head()

import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize

"""Sentence Extractions"""

def cleansentence(input):
   try: 
        r=re.match('[a-zA-Z]+', input)
   except:
       return None
   if r==None:
      return None
   else:
      return input

import re
def getsentences(revdf):
   l=len(revdf)
   papermeta = []
   revmeta = []
   revsentences = []
   print(len(revsentences))
   for i in range(l):
     papermeta.append([i, i*3, i*3+2])
     r1 = len(revsentences)
     try:
        r1sentences = sent_tokenize(df['r1'][i])
     except:
         r1sentences = ['No Review Available']
     for j in range(len(r1sentences)):
       sentences=cleansentence(r1sentences[j]) 
       if sentences != None:
         revsentences.append(sentences)
     
     r1len=len(revsentences)
     revmeta.append([i*3, r1, r1len-1])
     

     r2 = len(revsentences)
     try:
        r2sentences = sent_tokenize(df['r1'][i])
     except:
        r2sentences = ['No Review Available']
     for k in range(len(r2sentences)):
       sentences=cleansentence(r2sentences[k]) 
       if sentences != None:
         revsentences.append(sentences)
     
     
     r2len=len(revsentences)
     revmeta.append([i*3+1, r2, r2len-1])
     
     
     r3 = len(revsentences)
     try:
        r3sentences = sent_tokenize(df['r1'][i])
     except:
        r3sentences = ['No Review Available']
     for m in range(len(r3sentences)):
       sentences=cleansentence(r3sentences[m]) 
       if sentences != None:
         revsentences.append(sentences)
     
     r3len=len(revsentences)
     revmeta.append([i*3+2, r3, r3len-1])
   return papermeta, revmeta, revsentences

papermeta, revmeta, revsentences = getsentences(df)

revsen = revsentences

import pandas as pd 
papermeta = pd.DataFrame(papermeta)
papermeta.columns=['paperId','RevIdStart','RevIdEnd'] 
papermeta.head()
pd.DataFrame(papermeta).to_csv(data_path + "papermeta.csv")

revmeta = pd.DataFrame(revmeta)
revmeta.columns=['revId','SentenceStart','SentenceEnd']
pd.DataFrame(revmeta).to_csv(data_path+"Revmeta.csv")

revsentences = pd.DataFrame(revsentences)
revsentences.columns=['Text']
pd.DataFrame(revsentences).to_csv(data_path+"RevSentences.csv")


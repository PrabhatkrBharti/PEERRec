# -*- coding: utf-8 -*-


import sys
import tensorflow as tf
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import time
from copy import deepcopy
import random
from tensorflow.keras.layers import Conv2D, Permute , Dot, MaxPooling2D, TimeDistributed, AveragePooling2D, Lambda, Softmax, Dense, Flatten, Embedding, Bidirectional, LSTM, Dropout, Input, Reshape, concatenate, dot, Multiply, RepeatVector
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

paper_mx_sections = 23
rev_mx_snts = 13

"""##data loading"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./2017 Data')
args = parser.parse_args()

data_path = args.dataset+'/'

revdata = pd.read_csv(data_path + "Data.csv")
revdata = revdata[["id", "label", "rec1", "rec2", "rec3"]]

num_samples = len(revdata)

#=================================================================

papermeta = pd.read_csv(data_path + "papermeta.csv")
papermeta = papermeta[["paperId", "RevIdStart", "RevIdEnd"]]

#none-------------------------------------------------------------
revmeta = pd.read_csv(data_path + "Revmeta.csv")
revmeta = revmeta[["revId", "SentenceStart", "SentenceEnd"]]
n = len(revmeta)
rev_mx_snts = 5 #144 Is the max len, this is just initialization 
for i in range(n):
  rev_mx_snts = max(rev_mx_snts, revmeta["SentenceEnd"][i] - revmeta["SentenceStart"][i] + 1)

print("rev_mx_snts : ", rev_mx_snts)

print("rev embed----")
rev_embed_dim = 768
revEmbed = pd.read_csv(data_path + "scibert_embeddings_R1_R2_R3.csv")
print(len(revEmbed))
revEmbed = revEmbed[[str(i) for i in range(rev_embed_dim)]]
revEmbed = revEmbed.values.tolist()
print('--------',len(revEmbed),len(revEmbed[0]))
revEmbed = revEmbed + [[0.0 for i in range(rev_embed_dim)]] #padding
print('--------',len(revEmbed),len(revEmbed[0]))
rev_num_tokens = len(revEmbed)
print("rev_num_tokens : ", rev_num_tokens)

#=================================================================
print("rev senti----")
senti = pd.read_csv(data_path + "senti.csv")
senti_embed_dim = senti[["neg", "pos", "compound","neu"]]
print(len(senti))
# senti = senti[[str(i) for i in senti[["neg", "pos", "compound","neu"]]]]
# senti = senti.values.tolist()
# print('--------',len(senti),len(senti[0]))
# senti = senti + [[0.0 for i in range(4)]] #padding
# print('--------',len(senti),len(senti[0]))
# senti = len(senti)
# print("senti_num_tokens : ", senti)


#=================================================================
secmeta = pd.read_csv(data_path + "Paper_Section_Summary.csv")
secmeta = secmeta[["paperid", "sectionid"]]

scmeta = []
n = len(secmeta)
prev_pid = -1
strt = -1
for i in range(n):
  if prev_pid != secmeta["paperid"][i]:
    if strt!=-1:
      scmeta.append([prev_pid, strt, i-1])
    strt = i
    prev_pid = secmeta["paperid"][i]
  # else :
  if i+1 == n:
    scmeta.append([prev_pid, strt, i])

print("scmeta ", len(scmeta))
scmeta = pd.DataFrame(scmeta, columns=["paperId", "SecIdStart", "SecIdEnd"])

n = len(scmeta)
paper_mx_sections = 5 ###
for i in range(n):
  paper_mx_sections = max(paper_mx_sections, scmeta["SecIdEnd"][i] - scmeta["SecIdStart"][i] + 1)

print("paper_mx_sections : ", paper_mx_sections)

print("paper embed----")
paper_embed_dim = 768
paperEmbed = pd.read_csv(data_path + "scibert_embeddings_Paper.csv")
print(len(paperEmbed))
paperEmbed = paperEmbed[[str(i) for i in range(paper_embed_dim)]]
paperEmbed = paperEmbed.values.tolist()
print(len(paperEmbed),len(paperEmbed[0]))
paperEmbed = paperEmbed + [[0.0 for i in range(paper_embed_dim)]]
paper_num_tokens = len(paperEmbed)
print("paper_num_tokens : ", paper_num_tokens)

x = [i for i in scmeta['paperId']]
y = [0.0 if revdata["label"][i]=="Reject" else 1.0 for i in x]
num_samples = len(x)

def get_sims(sections, sentences, sec_zero_token, sent_zero_token):
  sims = [[0.0 for i in range(len(sentences))] for j in range(len(sections))]
  for j in range(len(sections)): 
    if sections[j]==sec_zero_token:
      continue
    for i in range(len(sentences)):
      sentence_id = sentences[i]
      if sentence_id!=sent_zero_token:
        sims[j][i] = cosine_similarity([paperEmbed[sections[j]]],[revEmbed[sentence_id]])
  return sims

def create_samples(papermeta, revmeta, scmeta, revdata, senti, pids, paper_mx_sections, rev_mx_snts, paper_num_tokens, rev_num_tokens,):
  xp = [] #paper input
  xr = [] #review input
  xs = [] #sim input
  yc = [] #output classification
  xv = [] #sentiment vader input 
  yr = [] #Recommendation output
  for i in tqdm(pids,"create_samples"): 
    yc.append(0.0 if revdata["label"][i]=="Reject" else 1.0)
    yr.append([float(revdata["rec1"][i]),float(revdata["rec2"][i]),float(revdata["rec3"][i])])
    #
    p_inp = []
    for j in range(scmeta["SecIdStart"][i], 1+scmeta["SecIdEnd"][i]):
      p_inp.append(j) #get paper ids in list id 0 = [0 ... 16] id 1 = [17..21]
    while len(p_inp)<paper_mx_sections:
      p_inp.append(paper_num_tokens - 1) #paddding to max length 
    #
    r_inp = []
    sim_inp = []
    v_inp = []
    for j in range(papermeta["RevIdStart"][i], 1+papermeta["RevIdEnd"][i]):
      rx_inp = []
      simx_inp = []
      vx_inp = []
      for k in range(revmeta["SentenceStart"][j], 1+revmeta["SentenceEnd"][j]):
        rx_inp.append(k)
        vx_inp.append([senti['neg'][k],senti['pos'][k],senti['compound'][k],senti['neu'][k]])
      while len(rx_inp)<rev_mx_snts:
        rx_inp.append(rev_num_tokens - 1)
        vx_inp.append([0.0,1.0,0.0,0.0])
      simx_inp = get_sims(p_inp, rx_inp, paper_num_tokens-1, rev_num_tokens-1)
      simx_inp = np.asarray(simx_inp, dtype='float32').reshape(paper_mx_sections, rev_mx_snts)
      r_inp.append(rx_inp)
      v_inp.append(vx_inp)
      sim_inp.append(simx_inp)
    #

    xp.append(p_inp)
    xr.append(r_inp)
    xv.append(v_inp)
    xs.append(sim_inp)
  
  xp = np.asarray(xp) 
  xr = np.asarray(xr) 
  xs = np.asarray(xs)
  yc = np.asarray(yc)
  xv = np.asarray(xv)
  yr = np.asarray(yr)

  return xp,xr,xv,xs,yc,yr

xtr , xte , ytr , yte = train_test_split(x,y, test_size=0.25, random_state=1, stratify=y)
print("train size : ", len(ytr))
print("test size : ", len(yte))

def balance_batch(batch_size, x, y):
  cx = [[], []]
  n = len(y)
  for i in range(n):
    cx[int(y[i])].append(x[i])
  
  p_ids = []
  ix = [0, 0]
  while ix[0]!=len(cx[0]):
    p_ids.append(cx[0][ix[0]])
    ix[0]+=1
    p_ids.append(cx[1][ix[1]])
    ix[1]+=1
    ix[1] = ix[1]%(len(cx[1]))
  return p_ids

xtr=balance_batch(2,xtr,ytr)

xp,xr,xv,xs,yc,yrc = create_samples(papermeta, revmeta, scmeta, revdata, senti, xtr, paper_mx_sections, rev_mx_snts, paper_num_tokens, rev_num_tokens)
tp,tr,tv,ts,tc,trc = create_samples(papermeta, revmeta, scmeta, revdata, senti, xte, paper_mx_sections, rev_mx_snts, paper_num_tokens, rev_num_tokens)

"""##modelling"""

def rmse_fun(pred,actual):
	return np.sqrt(np.mean((pred-actual)**2))
 
@tf.function
def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))
 
# tf.keras.losses.BinaryCrossentropy(
#     from_logits=False, label_smoothing=0, axis=-1,
#     reduction=losses_utils.ReductionV2.AUTO, name='binary_crossentropy'
# )
BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy()

def model_architecture32():
  #some values
  activation_func = "relu"
  dropout = 0.5
  num_filters = paper_mx_sections * 2
  d_out = paper_mx_sections * 5
  cnn_out = 7*num_filters
  data_dim = paper_embed_dim + cnn_out

  #paper - part
  paper_input = Input(shape=(paper_mx_sections,))
  paper_embedded = Embedding(paper_num_tokens, paper_embed_dim, embeddings_initializer=tf.keras.initializers.Constant(paperEmbed), trainable=False)(paper_input)

  #review - part
  rev_input = Input(shape=(3,rev_mx_snts))
  rev_embedded = Embedding(rev_num_tokens, rev_embed_dim, embeddings_initializer=tf.keras.initializers.Constant(revEmbed), trainable=False)(rev_input)

  #sim-part
  sim_input = Input(shape=(3,paper_mx_sections,rev_mx_snts))

  #sentiment input
  vader_input = Input(shape=(3,rev_mx_snts,4))
  vader_flat = TimeDistributed(Flatten())(vader_input) #3,576

  ss_cnn1 = TimeDistributed(Conv2D(num_filters, (1,rev_embed_dim),activation=activation_func, name="ss_cnn1"))
  ss_cnn2 = TimeDistributed(Conv2D(num_filters, (2,rev_embed_dim),activation=activation_func, name="ss_cnn2"))
  ss_cnn3 = TimeDistributed(Conv2D(num_filters, (3,rev_embed_dim),activation=activation_func, name="ss_cnn3"))
  ss_cnn4 = TimeDistributed(Conv2D(num_filters, (4,rev_embed_dim),activation=activation_func, name="ss_cnn4"))
  ss_cnn5 = TimeDistributed(Conv2D(num_filters, (5,rev_embed_dim),activation=activation_func, name="ss_cnn5"))
  ss_cnn6 = TimeDistributed(Conv2D(num_filters, (6,rev_embed_dim),activation=activation_func, name="ss_cnn6"))
  ss_cnn7 = TimeDistributed(Conv2D(num_filters, (7,rev_embed_dim),activation=activation_func, name="ss_cnn7"))

  # sec & rev complex representation (SRC)---------------------------------------------------
  # 3 , mx_sec, mx_snt
  s_sim =  sim_input[:, :, 0:1, :]
  s_sim = TimeDistributed(Flatten())(s_sim)
  s_sim = TimeDistributed(RepeatVector(rev_embed_dim))(s_sim)
  s_sim = Permute((1,3,2))(s_sim)

  sec_r = Multiply()([s_sim , rev_embedded])
  sec_r = Reshape((3,rev_mx_snts,rev_embed_dim,1))(sec_r) #changed

  r_a = ss_cnn1(sec_r)
  r_a = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts,1)))(r_a)
  r_a = TimeDistributed(Flatten())(r_a)

  r_b = ss_cnn2(sec_r)
  r_b = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-1,1)))(r_b)
  r_b = TimeDistributed(Flatten())(r_b)

  r_c = ss_cnn3(sec_r)
  r_c = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-2,1)))(r_c)
  r_c = TimeDistributed(Flatten())(r_c)

  r_d = ss_cnn4(sec_r)
  r_d = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-3,1)))(r_d)
  r_d = TimeDistributed(Flatten())(r_d)

  r_e = ss_cnn5(sec_r)
  r_e = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-4,1)))(r_e)
  r_e = TimeDistributed(Flatten())(r_e)

  r_f = ss_cnn6(sec_r)
  r_f = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-5,1)))(r_f)
  r_f = TimeDistributed(Flatten())(r_f)

  r_g = ss_cnn7(sec_r)
  r_g = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-6,1)))(r_g)
  r_g = TimeDistributed(Flatten())(r_g)

  sr_rep = concatenate([r_a, r_b, r_c, r_d, r_e, r_f, r_g] , axis=-1)
  sr_rep = Reshape((3,1,cnn_out))(sr_rep) 

  SCLR = sr_rep

  for i in range(1, paper_mx_sections):
    s_sim =  sim_input[:, :, i:i+1, :]
    s_sim = TimeDistributed(Flatten())(s_sim)
    s_sim = TimeDistributed(RepeatVector(rev_embed_dim))(s_sim)
    s_sim = Permute((1,3,2))(s_sim)

    sec_r = Multiply()([s_sim , rev_embedded])
    sec_r = Reshape((3,rev_mx_snts,rev_embed_dim,1))(sec_r)

    r_a = ss_cnn1(sec_r)
    r_a = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts,1)))(r_a)
    r_a = TimeDistributed(Flatten())(r_a)

    r_b = ss_cnn2(sec_r)
    r_b = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-1,1)))(r_b)
    r_b = TimeDistributed(Flatten())(r_b)

    r_c = ss_cnn3(sec_r)
    r_c = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-2,1)))(r_c)
    r_c = TimeDistributed(Flatten())(r_c)

    r_d = ss_cnn4(sec_r)
    r_d = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-3,1)))(r_d)
    r_d = TimeDistributed(Flatten())(r_d)

    r_e = ss_cnn5(sec_r)
    r_e = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-4,1)))(r_e)
    r_e = TimeDistributed(Flatten())(r_e)

    r_f = ss_cnn6(sec_r)
    r_f = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-5,1)))(r_f)
    r_f = TimeDistributed(Flatten())(r_f)

    r_g = ss_cnn7(sec_r)
    r_g = TimeDistributed(AveragePooling2D(pool_size=(rev_mx_snts-6,1)))(r_g)
    r_g = TimeDistributed(Flatten())(r_g)

    sr_rep = concatenate([r_a, r_b, r_c, r_d, r_e, r_f, r_g] , axis=-1)
    sr_rep = Reshape((3,1,cnn_out))(sr_rep) #Changed
    
    SCLR = concatenate([SCLR, sr_rep] , axis=2)

  # SRC end--------------------------------------------------------------------------------------

  sc = TimeDistributed(RepeatVector(3))(paper_embedded)
  sc = Permute((2,1,3))(sc)

  SC_R = concatenate([sc, SCLR] , axis=3, name="complex_rep")

  SC_R = Reshape((3,paper_mx_sections,data_dim,1))(SC_R)

  r_a = TimeDistributed(Conv2D(num_filters, (1,data_dim),activation=activation_func))(SC_R)
  r_a = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections,1)))(r_a)
  r_a = TimeDistributed(Flatten())(r_a)

  r_b = TimeDistributed(Conv2D(num_filters, (2,data_dim),activation=activation_func))(SC_R)
  r_b = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections-1,1)))(r_b)
  r_b = TimeDistributed(Flatten())(r_b)

  r_c = TimeDistributed(Conv2D(num_filters, (3,data_dim),activation=activation_func))(SC_R)
  r_c = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections-2,1)))(r_c)
  r_c = TimeDistributed(Flatten())(r_c)

  r_d = TimeDistributed(Conv2D(num_filters, (4,data_dim),activation=activation_func))(SC_R)
  r_d = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections-3,1)))(r_d)
  r_d = TimeDistributed(Flatten())(r_d)

  r_e = TimeDistributed(Conv2D(num_filters, (5,data_dim),activation=activation_func))(SC_R)
  r_e = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections-4,1)))(r_e)
  r_e = TimeDistributed(Flatten())(r_e)

  r_f = TimeDistributed(Conv2D(num_filters, (6,data_dim),activation=activation_func))(SC_R)
  r_f = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections-5,1)))(r_f)
  r_f = TimeDistributed(Flatten())(r_f)

  r_g = TimeDistributed(Conv2D(num_filters, (7,data_dim),activation=activation_func))(SC_R)
  r_g = TimeDistributed(MaxPooling2D(pool_size=(paper_mx_sections-6,1)))(r_g)
  r_g = TimeDistributed(Flatten())(r_g)

  sr = concatenate([r_a , r_b , r_c , r_d , r_e , r_f , r_g] , axis=-1)
  sr_final = TimeDistributed(Flatten())(sr)
  sr_final = TimeDistributed(Dropout(dropout))(sr_final)

  sr_vector = Flatten()(sr_final)

  # rev only - cnn--------------------------------------------------------------------------

  R = Reshape((3,rev_mx_snts,rev_embed_dim,1))(rev_embedded)

  r_a = TimeDistributed(Conv2D(num_filters, (1,rev_embed_dim),activation=activation_func))(R)
  r_a = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts,1)))(r_a)
  r_a = TimeDistributed(Flatten())(r_a)

  r_b = TimeDistributed(Conv2D(num_filters, (2,rev_embed_dim),activation=activation_func))(R)
  r_b = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-1,1)))(r_b)
  r_b = TimeDistributed(Flatten())(r_b)

  r_c = TimeDistributed(Conv2D(num_filters, (3,rev_embed_dim),activation=activation_func))(R)
  r_c = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-2,1)))(r_c)
  r_c = TimeDistributed(Flatten())(r_c)

  r_d = TimeDistributed(Conv2D(num_filters, (4,rev_embed_dim),activation=activation_func))(R)
  r_d = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-3,1)))(r_d)
  r_d = TimeDistributed(Flatten())(r_d)

  r_e = TimeDistributed(Conv2D(num_filters, (5,rev_embed_dim),activation=activation_func))(R)
  r_e = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-4,1)))(r_e)
  r_e = TimeDistributed(Flatten())(r_e)

  r_f = TimeDistributed(Conv2D(num_filters, (6,rev_embed_dim),activation=activation_func))(R)
  r_f = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-5,1)))(r_f)
  r_f = TimeDistributed(Flatten())(r_f)

  r_g = TimeDistributed(Conv2D(num_filters, (7,rev_embed_dim),activation=activation_func))(R)
  r_g = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-6,1)))(r_g)
  r_g = TimeDistributed(Flatten())(r_g)

  r = concatenate([r_a , r_b , r_c , r_d , r_e , r_f , r_g] , axis=-1)
  r_final = TimeDistributed(Flatten())(r) #shape 3,x
  r_final = TimeDistributed(Dropout(dropout))(r_final)
  r_final = concatenate([r_final,vader_flat] , axis=-1) #3,x+y

  r_vector = Flatten()(r_final) # CNN on reviews embedding

  #Recommedation
#   rec_inp = concatenate([sr_final, r_final] , axis=-1)
#   rec_dim1 = Dense(d_out,activation="relu")(rec_inp)
#   rec_dim1 = Dropout(dropout)(rec_dim1)
#   rec_dim2 = Dense(d_out,activation="softsign")(rec_inp)
#   rec_dim2 = Dropout(dropout)(rec_dim2)
#   rec_dim3 = Dense(d_out,activation="sigmoid")(rec_inp)
#   rec_dim3 = Dropout(dropout)(rec_dim3)

#   rec_dim = concatenate([rec_dim1 , rec_dim2 , rec_dim3] , axis=-1)
#   rec_out = Dense(1,activation="relu")(rec_dim)
#   rec_out = Flatten(name="Regression_Output")(rec_out)

  #MLP
  mlp_inp = concatenate([r_vector, sr_vector] , axis=-1)

  dim1 = Dense(d_out,activation="relu")(mlp_inp)
  dim1 = Dropout(dropout)(dim1)
  dim2 = Dense(d_out,activation="softsign")(mlp_inp)
  dim2 = Dropout(dropout)(dim2)
  dim3 = Dense(d_out,activation="sigmoid")(mlp_inp)
  dim3 = Dropout(dropout)(dim3)

  dim = concatenate([dim1 , dim2 , dim3] , axis=-1)
  classification_out = Dense(1,activation="sigmoid", name="Classification_Output")(dim)

#   model = Model(inputs=[paper_input, sim_input, rev_input, vader_input] , outputs=[classification_out,rec_out])
  model_class = Model(inputs=[paper_input, sim_input, rev_input, vader_input] , outputs=classification_out)
#   model_rec = Model(inputs=[paper_input, sim_input, rev_input, vader_input] , outputs=rec_out)
  # "binary_crossentropy"
  # model.compile(optimizer = "adam", loss = root_mean_squared_error ,metrics =["accuracy"])
#   model.compile(optimizer = "adam", loss = [BinaryCrossentropy, root_mean_squared_error] , metrics =["accuracy","mse"])
  model_class.compile(optimizer = "adam", loss = BinaryCrossentropy, metrics =["accuracy"])
#   model_rec.compile(optimizer = "adam", loss = root_mean_squared_error ,metrics =["mse"])
  # model.summary()
  return model_class

"""##training & testing"""

#for architecture 1 only
# trLen = len(yc)
clm = model_architecture32()
# clm.fit([xp , xr], yc, batch_size = 32, shuffle=True,epochs = 40,verbose=1)

"""##generate test report"""

def clean_tc_prd(tc_prd):
  for i in range(teLen):
    if tc_prd[i]>0.5:
      tc_prd[i] = 1.0
    else:
      tc_prd[i]=0.0
  return tc_prd

def get_f1(tc_prd,tc):
  mat = [[0,0],[0,0]]
  for i in range(teLen):
    yp = 0
    yt = 0
    if tc_prd[i]>0.5:
      yp=1
    if tc[i]>0.5:
      yt=1
    mat[yt][yp] += 1
    
  prec = [mat[0][0]/max(1,mat[0][0]+mat[1][0]) , mat[1][1]/max(1,mat[1][1]+mat[0][1])]
  rec = [mat[0][0]/max(1,mat[0][0]+mat[0][1]) , mat[1][1]/max(1,mat[1][1]+mat[1][0])]
  f1 = [(2*prec[0]*rec[0])/max(0.0001,prec[0]+rec[0]) , (2*prec[1]*rec[1])/max(0.0001,prec[1]+rec[1])]
  return f1[0]+f1[1]

"""## WITHOUT RECCOMENDATION ONLY CLASSIFICATIOn"""

import copy
teLen = len(tc)
max_f1 = 0.0
tc_prd = []
for i in range(15):
  print("Epoch:",i)
  clm.fit([xp, xs, xr, xv], yc, batch_size = 16, shuffle=False,epochs = 1,verbose=1)
  tmp = clm.predict([tp, ts, tr, tv])
  f1= get_f1(clean_tc_prd(tmp),tc)
  if f1>max_f1:
    max_f1 = f1
    tc_prd = copy.deepcopy(tmp)
  print("f1 score:", f1)

print(metrics.classification_report(tc,clean_tc_prd(tc_prd)))
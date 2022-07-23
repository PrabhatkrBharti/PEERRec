# -*- coding: utf-8 -*-


import sys
import tensorflow as tf
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
from tensorflow.keras.layers import Conv2D, Bidirectional,Permute , Dot, MaxPooling2D, TimeDistributed, AveragePooling2D, Lambda, Softmax, Dense, Flatten, Embedding, Bidirectional, LSTM, Dropout, Input, Reshape, concatenate, dot, Multiply, RepeatVector, Activation
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import argparse
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.optimizers import Adam


from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error

seed = 9999
 
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

paper_mx_sections = 23
rev_mx_snts = 13

"""##data loading"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./2017 Data')
args = parser.parse_args()

data_path = args.dataset+'/'

"""actual data load"""

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

from sklearn.preprocessing import MinMaxScaler,StandardScaler
min_max=MinMaxScaler()
sc=StandardScaler()

def create_samples(papermeta, revmeta, scmeta, revdata, senti, pids, paper_mx_sections, rev_mx_snts, paper_num_tokens, rev_num_tokens,):
  xp = [] #paper input
  xr = [] #review input
  yc = [] #output classification
  xv = [] #sentiment vader input 
  yr = [] #Recommendation output
  for i in tqdm(pids,"create_samples"): 
    yc.append(0.0 if revdata["label"][i]=="Reject" else 1.0)
    yr.append([float(revdata["rec1"][i]),float(revdata["rec2"][i]),float(revdata["rec3"][i])])
    p_inp = []
    p_rinp = []
    for j in range(scmeta["SecIdStart"][i], 1+scmeta["SecIdEnd"][i]):
      p_inp.append(j) #get paper ids in list id 0 = [0 ... 16] id 1 = [17..21]
      p_rinp.append(j)
    while len(p_inp)<paper_mx_sections:
      p_inp.append(paper_num_tokens - 1) #paddding to max length 
    #
    r_inp = []
    v_inp = []
    for j in range(papermeta["RevIdStart"][i], 1+papermeta["RevIdEnd"][i]):
      rx_inp = []
      vx_inp = []
      for k in range(revmeta["SentenceStart"][j], 1+revmeta["SentenceEnd"][j]):
        rx_inp.append(k)
        vx_inp.append([senti['neg'][k],senti['pos'][k],senti['compound'][k],senti['neu'][k]])
      while len(rx_inp)<rev_mx_snts:
        rx_inp.append(rev_num_tokens - 1)
        vx_inp.append([0.0,1.0,0.0,0.0])
      r_inp.append(rx_inp)
      v_inp.append(vx_inp)
    #
    xp.append(p_inp)
    xr.append(r_inp)
    xv.append(v_inp)
  
  xp = np.asarray(xp) 
  xr = np.asarray(xr) 
  yc = np.asarray(yc)
  xv = np.asarray(xv)
  yr = np.asarray(yr)

  xr = np.array(xr).reshape(-1,3*rev_mx_snts)
  min_max.fit_transform(xr)
  xr = np.array(xr).reshape(-1,3,rev_mx_snts)

  return xp,xr,xv,yc,yr

xtr, xte, ytr, yte = train_test_split(x,y, test_size=0.25, random_state=1, stratify=y)
print("train size : ", len(ytr))
print("test size : ", len(yte))

def balance_batch(batch_size, x, y):
  cx = [[], []]
  n = len(y)
  for i in range(n):
    cx[int(y[i])].append(x[i])
  random.shuffle(cx[0])
  random.shuffle(cx[1])
  p_ids = []
  ix = [0, 0]
  while ix[0]!=len(cx[0]):
    p_ids.append(cx[0][ix[0]])
    ix[0]+=1
    p_ids.append(cx[1][ix[1]])
    ix[1]+=1
    ix[1] = ix[1]%(len(cx[1]))
  return p_ids

xtr=balance_batch(-1,xtr,ytr)

xp,xr,xv,yc,yrc = create_samples(papermeta, revmeta, scmeta, revdata, senti, xtr, paper_mx_sections, rev_mx_snts, paper_num_tokens, rev_num_tokens)
tp,tr,tv,tc,trc = create_samples(papermeta, revmeta, scmeta, revdata, senti, xte, paper_mx_sections, rev_mx_snts, paper_num_tokens, rev_num_tokens)

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

activation_func = "relu"
dropout = 0.7
rec_out = paper_mx_sections + rev_mx_snts
c_out = paper_mx_sections + 2*rev_mx_snts
paper_num_filters = max(28, paper_mx_sections)
rev_num_filters = max(48, rev_mx_snts // 3)
d_out = paper_mx_sections + rev_mx_snts
num_filters = d_out // 3
units = 200

"""### ### Model New Propoosed Model"""

attention_model = None
def model_architectureNPM():  
  #paper - part
  paper_input = Input(shape=(paper_mx_sections,))
  paper_embedded = Embedding(paper_num_tokens, paper_embed_dim, embeddings_initializer=tf.keras.initializers.Constant(paperEmbed), trainable=False)(paper_input)

  #review - part
  rev_input = Input(shape=(3,rev_mx_snts))
  rev_embedded = Embedding(rev_num_tokens, rev_embed_dim, embeddings_initializer=tf.keras.initializers.Constant(revEmbed), trainable=False)(rev_input)

  #sentiment input
  vader_input = Input(shape=(3,rev_mx_snts,4))
  vader_flat = TimeDistributed(Flatten())(vader_input) #3,576

  ##paper only - cnn
  SC = Reshape((paper_mx_sections,paper_embed_dim,1))(paper_embedded)

  p_a = Conv2D(paper_num_filters, (1,paper_embed_dim),activation=activation_func)(SC)
  p_a = MaxPooling2D(pool_size=(paper_mx_sections,1))(p_a)
  p_a = Flatten()(p_a)

  p_b = Conv2D(paper_num_filters, (2,paper_embed_dim),activation=activation_func)(SC)
  p_b = MaxPooling2D(pool_size=(paper_mx_sections-1,1))(p_b)
  p_b = Flatten()(p_b)

  p_c = Conv2D(paper_num_filters, (3,paper_embed_dim),activation=activation_func)(SC)
  p_c = MaxPooling2D(pool_size=(paper_mx_sections-2,1))(p_c)
  p_c = Flatten()(p_c)

  p_d = Conv2D(paper_num_filters, (4,paper_embed_dim),activation=activation_func)(SC)
  p_d = MaxPooling2D(pool_size=(paper_mx_sections-3,1))(p_d)
  p_d = Flatten()(p_d)

  p_e = Conv2D(paper_num_filters, (5,paper_embed_dim),activation=activation_func)(SC)
  p_e = MaxPooling2D(pool_size=(paper_mx_sections-4,1))(p_e)
  p_e = Flatten()(p_e)

  sc = concatenate([p_a , p_b , p_c , p_d , p_e] , axis=-1)
  sc_final = Flatten()(sc)
  sc_final = Dropout(dropout)(sc_final)

  sc_vector = Flatten()(sc_final)
  
  # Self Attention On Reviews -------------------------------------------------------------------------
  # activations = TimeDistributed(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(rev_embedded)
  activations = TimeDistributed(Bidirectional(LSTM(units, return_sequences=True)))(rev_embedded)

  # attention
  attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
  attention = TimeDistributed(Flatten())(attention)
  attention = TimeDistributed(Activation('softmax'), name="attention_vec")(attention)
  attention = TimeDistributed(RepeatVector(2*units))(attention)
  attention = TimeDistributed(Permute([2, 1]))(attention)

  representation_activations = Multiply()([activations, attention])
  # representation_activations = Lambda(lambda xin: K.sum(xin, axis=0))(representation_activations)
  representation_activations = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(representation_activations)
  activations = TimeDistributed(Flatten())(representation_activations)

  # rev only - cnn--------------------------------------------------------------------------
  
  R = Reshape((3,rev_mx_snts,rev_embed_dim,1))(rev_embedded)

  r_a = TimeDistributed(Conv2D(rev_num_filters, (1,rev_embed_dim),activation=activation_func), name="r_a")(R)
  r_a = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts,1)))(r_a)
  r_a = TimeDistributed(Flatten())(r_a)

  r_b = TimeDistributed(Conv2D(rev_num_filters, (2,rev_embed_dim),activation=activation_func))(R)
  r_b = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-1,1)))(r_b)
  r_b = TimeDistributed(Flatten())(r_b)

  r_c = TimeDistributed(Conv2D(rev_num_filters, (3,rev_embed_dim),activation=activation_func))(R)
  r_c = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-2,1)))(r_c)
  r_c = TimeDistributed(Flatten())(r_c)

  r_d = TimeDistributed(Conv2D(rev_num_filters, (4,rev_embed_dim),activation=activation_func))(R)
  r_d = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-3,1)))(r_d)
  r_d = TimeDistributed(Flatten())(r_d)

  r_e = TimeDistributed(Conv2D(rev_num_filters, (5,rev_embed_dim),activation=activation_func))(R)
  r_e = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-4,1)))(r_e)
  r_e = TimeDistributed(Flatten())(r_e)

  r_f = TimeDistributed(Conv2D(rev_num_filters, (6,rev_embed_dim),activation=activation_func))(R)
  r_f = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-5,1)))(r_f)
  r_f = TimeDistributed(Flatten())(r_f)

  r_g = TimeDistributed(Conv2D(rev_num_filters, (7,rev_embed_dim),activation=activation_func))(R)
  r_g = TimeDistributed(MaxPooling2D(pool_size=(rev_mx_snts-6,1)))(r_g)
  r_g = TimeDistributed(Flatten())(r_g)

  r = concatenate([r_a , r_b , r_c , r_d , r_e , r_f , r_g] , axis=-1)
  r_final = TimeDistributed(Flatten())(r) #shape 3,x
  r_final = TimeDistributed(Dropout(dropout))(r_final)
  r_final = concatenate([r_final,vader_flat,activations] , axis=-1) #3,x+y

  r_vector = Flatten()(r_final) # CNN on reviews embedding

  #Recommedation
  sc_final = RepeatVector(3)(sc_vector)
  rec_inp = concatenate([r_final, sc_final] , axis=-1)
  rec_dim1 = Dense(d_out,activation="relu")(rec_inp)
  rec_dim1 = Dropout(dropout)(rec_dim1)
  rec_dim2 = Dense(d_out,activation="softsign")(rec_inp)
  rec_dim2 = Dropout(dropout)(rec_dim2)
  rec_dim3 = Dense(d_out,activation="sigmoid")(rec_inp)
  rec_dim3 = Dropout(dropout)(rec_dim3)

  rec_dim = concatenate([rec_dim1 , rec_dim2 , rec_dim3] , axis=-1)
  rec_out = Dense(1,activation="relu")(rec_dim)
  rec_out = Flatten(name="Regression_Output")(rec_out)

  #MLP
  mlp_inp = concatenate([r_vector, sc_vector] , axis=-1)

  dim1 = Dense(d_out,activation="relu")(mlp_inp)
  dim1 = Dropout(dropout)(dim1)
  dim2 = Dense(d_out,activation="softsign")(mlp_inp)
  dim2 = Dropout(dropout)(dim2)
  dim3 = Dense(d_out,activation="sigmoid")(mlp_inp)
  dim3 = Dropout(dropout)(dim3)

  dim = concatenate([dim1 , dim2 , dim3] , axis=-1)
  classification_out = Dense(1,activation="sigmoid", name="Classification_Output")(dim)

  model = Model(inputs=[paper_input, rev_input, vader_input] , outputs=[classification_out,rec_out])
  model.compile(optimizer = "adam", loss = [BinaryCrossentropy, root_mean_squared_error] ,metrics =["accuracy","mse"])

  cls_model = Model(inputs=[paper_input, rev_input, vader_input] , outputs=classification_out)
  cls_model.compile(optimizer = "adam", loss = BinaryCrossentropy ,metrics =["accuracy"])

  rec_model = Model(inputs=[paper_input, rev_input, vader_input] , outputs=rec_out)
  rec_model.compile(optimizer = "adam", loss = root_mean_squared_error ,metrics =["mse"])
  
  global attention_model
  attention_model = Model(inputs=[paper_input, rev_input, vader_input] , outputs=model.get_layer('attention_vec').output)

  return cls_model,rec_model,model

class attention_class(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(attention_class,self).__init__(**kwargs)
 
    def build(self,input):
        self.W=self.add_weight(name='attention_weight', shape=(input[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention_class, self).build(input)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
    def compute_output_shape(self, input_shape):
        return (input_shape[0])

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

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

trLen = len(yc)
_,clm,_ = model_architectureNPM()
train_inputs = [xp, xr, xv]
test_inputs = [tp, tr, tv]

#With Recom Addded With Reccomendation RMSE MAX:
import copy
teLen = len(tc)
min_rmse = 18.0
trc_prd = []
attention_values = None
for i in range(48):
  print("Epoch:",i)
  clm.fit(train_inputs, yrc, batch_size = 32, shuffle=False,epochs = 1,verbose=1)
  rec_tmp = clm.predict(test_inputs)
  RMSE=rmse_fun(rec_tmp,trc)
  if RMSE<min_rmse:
    min_rmse = RMSE
    trc_prd = copy.deepcopy(rec_tmp)
  print("RMSE:", RMSE)

print("Recommedation Score RMSE:",rmse_fun(trc_prd, trc))
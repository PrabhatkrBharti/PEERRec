# -*- coding: utf-8 -*-


import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./2017 Data')
args = parser.parse_args()

data_path = args.dataset+'/'


df = pd.read_csv(data_path + "RevSentences.csv", encoding='utf8')
df.head()

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def find_sentiment(tweet):
    return sia.polarity_scores(tweet)

vader_sentiments = df.Text.apply(find_sentiment)

df['vader-sentiment'] = vader_sentiments

x = df[['vader-sentiment']]

df['neg'] = df['vader-sentiment'].apply(lambda x: x.get('neg')).dropna()

df['pos'] = df['vader-sentiment'].apply(lambda x: x.get('pos')).dropna()

df['compound'] = df['vader-sentiment'].apply(lambda x: x.get('compound')).dropna()

df['neu'] = df['vader-sentiment'].apply(lambda x: x.get('neu')).dropna()

new = df[['neg','pos','compound','neu']]

new['rev_length'] = df['Text'].apply(lambda x : len(x))

df_2 = pd.read_csv(data_path+"Data.csv", encoding='utf8')
df1 = df_2[['r1' , 'rec1']]
df1.rename(columns = {'r1':'ReviewText' , 'rec1' : "Recommendation"}, inplace = True)
df2 = df_2[['r2' , 'rec2']]
df2.rename(columns = {'r2':'ReviewText' , 'rec2' : "Recommendation"}, inplace = True)
df3 = df_2[['r3' , 'rec3']]
df3.rename(columns = {'r3':'ReviewText' , 'rec3' : "Recommendation"}, inplace = True)
df1 = df1.append(df2 , ignore_index = True)
df_2 = df1.append(df3 ,ignore_index = True)
df_2.head()

vader_sentiments = df_2.ReviewText.apply(find_sentiment)

df_2['vader-sentiment'] = vader_sentiments

x = df_2[['vader-sentiment']]

df_2['neg'] = df_2['vader-sentiment'].apply(lambda x: x.get('neg')).dropna()

df_2['pos'] = df_2['vader-sentiment'].apply(lambda x: x.get('pos')).dropna()

df_2['compound'] = df_2['vader-sentiment'].apply(lambda x: x.get('compound')).dropna()

df_2['neu'] = df_2['vader-sentiment'].apply(lambda x: x.get('neu')).dropna()

new = df_2[['neg','pos','compound','neu']]
new['length'] = df_2['ReviewText'].apply(lambda x : len(x))
new['Rec'] = df_2['Recommendation']

import seaborn as sb
import matplotlib as mp
print(new.corr())
dataplot = sb.heatmap(new.corr(), cmap="YlGnBu", annot=True)

"""higher the score  more closely is the entity of 

*   List item
*   List item


"""

import pandas as pd 
pd.DataFrame(new).to_csv(data_path+"senti.csv")
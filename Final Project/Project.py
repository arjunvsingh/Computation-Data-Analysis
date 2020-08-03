# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:51:36 2020

@author: Arie
"""
import pandas as pd
import numpy as np
import ftfy
import re
import collections

df = pd.read_csv('C:/Users/Arie/Desktop/OMSA/4_ISYE6740/Project/senators.csv', header=0, encoding = 'latin1')
df['ratio'] = df['replies']/(df['favorites'] + 1)
df['logfavorites'] = np.log(df['favorites'] + 1)
df['party'] = df.apply(lambda x: "D" if x['party'] == "I" else x['party'], axis = 1)

GroupUser = df.groupby(['user'], as_index=False)['logfavorites'].agg(['min','max']).reset_index()
df = pd.merge(df,GroupUser, how = 'inner', on = 'user')
df['likenorm'] = (df['logfavorites'] - df['min'])/(df['max'] - df['min'])
df.drop(['min', 'max'], axis = 1, inplace = True)

df['Clean text'] = df['text'].apply(lambda x: ftfy.fix_text(x))

def find_mentioned(tweet):
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_hashtags(tweet):
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

# make new columns for mentioned usernames and hashtags
df['mentioned'] = df['Clean text'].apply(find_mentioned)
df['hashtags'] = df['Clean text'].apply(find_hashtags)

# Clean up text
df['Clean text'] = df['Clean text'].str.replace('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', '', case=False)
df['Clean text'] = df['Clean text'].str.replace('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', case=False)
df['Clean text'] = df['Clean text'].str.replace('http\S+|www.\S+',', ''', case=False)

df.drop(['created_at','text', 'url', 'replies','retweets','user','bioguide_id','state','mentioned','hashtags'], axis = 1, inplace = True)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from scipy.sparse import coo_matrix, hstack

tfidf = TfidfVectorizer()
df_tfidf = tfidf.fit_transform(df['Clean text'])
pos = np.array(df['likenorm'].tolist())
neg = df['ratio'].tolist()
size = len(pos)
df_tfidf = hstack([df_tfidf,np.reshape(np.array(pos),(size,1)),np.reshape(np.array(neg),(size,1))])

xtrain, xtest, ytrain, ytest = train_test_split(df_tfidf,df['party'].tolist(), test_size=0.2, random_state=40)

enc = LabelEncoder()
ytrain = enc.fit_transform(ytrain)
ytest = enc.fit_transform(ytest)

clf = MultinomialNB()

clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)


print("The accuracy of the Multinomial NB model is {:.2f}%".format(f1_score(ytest, pred, average="micro")*100))


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth = 25,random_state=40, n_estimators = 250, min_samples_split = 10).fit(xtrain,ytrain)
pred = clf.predict(xtest)

print("The accuracy of the Random Forest model is {:.2f}%".format(f1_score(ytest, pred, average="micro")*100))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1',random_state=40, solver='liblinear', C=4).fit(xtrain, ytrain)
pred = clf.predict(xtest)

print("The accuracy of the LASSO model is {:.2f}%".format(f1_score(ytest, pred, average="micro")*100))
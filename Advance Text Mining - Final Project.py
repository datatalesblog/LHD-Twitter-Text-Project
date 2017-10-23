
# coding: utf-8

# In[55]:

import numpy as np
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as pt
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm
import re
import string

raw_text = []
raw_score = []
raw_text2 = []
raw_text3 = []

#inputing data
with open('C:\\Users\\Amit Bhalerao\\Documents\\CareerCenter\\DataTales\\LHDTwitter\\RawData.csv') as file1:
    raw_data = csv.reader(file1)
    for cols in raw_data:
        raw_text.append(cols[0])
        raw_score.append(cols[1])
        
#cleaning - removing links and digits    
for i in range(len(raw_text)):
    raw_text3.append(re.sub(r'http\S+', " ", raw_text[i]))
    
for i in range(len(raw_text3)):
    raw_text2.append(re.sub("\d+", " ", raw_text3[i]))

#generating TfIdf vector matrix    
tfi = TfidfVectorizer(stop_words = {'a','the','an','of','for','we','are','was','is','were','or','also','am','as','can','by','co','his','if','how','into','it','ll','too','this','these','those','they','yours','yourself'},max_df = 0.099,min_df = 0.001)
tfidf = tfi.fit_transform(raw_text2)
names = tfi.get_feature_names()
tfidf = tfidf.toarray()
tfidf = pd.DataFrame(tfidf)
tfidf.columns = names
score = pd.DataFrame(raw_score)

#linear regression using tfidf and retweet count
linreg = LinearRegression()
linreg.fit(tfidf, score)
lin_pred_score = linreg.predict(tfidf)
raw_scr = np.asarray(score, dtype=float)
r2 = metrics.r2_score(raw_scr,lin_pred_score)

#saving the coefficients in an arra
coefs = np.array(linreg.coef_)
coefs = coefs.flatten()

#outputting the tokens and coefficients in a dictionary
coef_dict = {}
for i in range(len(names)):
    coef_dict[names[i]] = coefs[i]

#exporting results in a csv
w = csv.writer(open("coef_output.csv", "w"))
for key, val in coef_dict.items():
    w.writerow([key, val])


# In[ ]:




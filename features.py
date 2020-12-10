""" The following requirements must be satisfied to run the functions 

!pip install polyglot
!pip install PyICU
!pip install pycld2
!pip install Morfessor
!polyglot download sentiment2.en

"""

import numpy as np
import nltk
from nltk.util import ngrams
from polyglot.text import Word
from polyglot.text import Text

def external_features(headline, body):

  train_headline_sentences = []
  train_body_sentences = []
  train_headline_sentences.append(headline)
  train_body_sentences.append(body)

  ext = []
  i = 0
  for sent1,sent2 in zip(train_headline_sentences,train_body_sentences):
    print(i)
    i+=1
    vec = []
    #character ngrams
    for n in range(2,17):
      n_grams_1 = ngrams(sent1.lower(), n)
      n_grams_2 = ngrams(sent2.lower(),n)
      vec.append(len(list(set(n_grams_1).intersection(n_grams_2))))

    #word ngrams
    for n in range(2,7):
      n_grams_1 = ngrams(sent1.lower().split(), n)
      n_grams_2 = ngrams(sent2.lower().split(),n)
      vec.append(len(list(set(n_grams_1).intersection(n_grams_2))))

    #Sentence polarity
    flag=False
    text1 = Text(sent1)
    text2 = Text(sent2)
    pol1 = 0
    pol2 = 0
    for word in text1.words:
      try:
        pol1+=word.polarity
      except:
        flag=True
        vec.append(0)
        break
    if (flag==True):
      ext.append(vec)
      continue
    flag=False
    for word in text2.words:
      try:
        pol2+=word.polarity
      except:
        flag=True
        vec.append(0)
        break
    if (flag==True):
      ext.append(vec)
      continue
    
    
    pol1 = pol1/(len(sent1.split())*1.0)
    pol2 = pol2/(len(sent2.split())*1.0)
    print("pol1 : ", pol1)
    print("pol2 : ", pol2)
    print(pol1-pol2)
    vec.append(pol1-pol2)

    ext.append(vec)

  return np.array(ext)

from sklearn.feature_extraction.text import TfidfVectorizer

def statistical_features(h, b):

  v1 = TfidfVectorizer(max_features= 2261)
  v2 = TfidfVectorizer(max_features= (10000- 2261))
  headline = []
  body = []
  headline.append(h)
  body.append(b)
  statistical_h = v1.fit_transform(headline)
  statistical_b = v1.fit_transform(body)
  #print(statistical_h.shape)
  a = statistical_h.toarray()
  #a.shape[1]
  b = np.zeros((1, 2261 - a.shape[1]))  
  #b.shape
  statistical_h = np.append(a, b).reshape((1, 2261))
  print(statistical_h.shape)
  print(statistical_b.shape)
  c = statistical_b.toarray()
  #c.shape[1]
  d = np.zeros((1, 10000 - 2261 - c.shape[1]))
  #d.shape
  statistical_b = np.append(c, d).reshape((1, 10000-2261))
  #print(statistical_b.shape)
  final_statistical_features = np.concatenate((statistical_h, statistical_b),axis = 1)
  print(final_statistical_features.shape)

  return final_statistical_features
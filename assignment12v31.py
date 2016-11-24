# IS620 - Assignment 12 (6.10.3)
# Program: assignment12.py
# Student: Neil Acampa
# Date:    10/31/16
# Function:

#    6.10.3   Word sense disambiquation
#    Prompt for keyword
#    Read Alice and wonderland and parse sentences with keyword
#    Store in sentword array
#    Find keyword in sentence with 2 prior words and 2 words after keyword
#    Store in sentwordadj array
#
#    Using the keyword and the wordnet update the syns and synsdef arrrays
#
#    Create features using the Lesk algorithm
#    For a given word and sentence:                                                   
#    The Lesk algorithm returns a Synset with the highest number of overlapping words
#    between the context sentence and different definitions from each Synset. 
#    The feature set shows the Snyset and Synset definition   
#    Display the features   
#
#    Evaluate features pos/neg score


from __future__ import absolute_import 
from __future__ import division
import re
import os 
import math
import decimal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import networkx as nx
import random
from urllib import urlopen
import nltk
nltk.download('gutenberg')
from nltk import word_tokenize
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('senseval')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('lesk')
nltk.download('stopwords')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import senseval
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import sentiwordnet as swn




linelst=[]
lines  = ""
allwords          = []   # Contains all words

 
sentword          = []   # Holds an entire sentence with comma separted words
sentwordadj       = []   # Holds a sentence with the 2 prior and 2 words after the keyword

syns              = []   # Synset id for a specific word 
synsdef           = []   # The corresponding Synset Sentence
synsname          = []   # The corresponding Synset name (i.e. 'look.v.09')

results           = []
# Table Elements

fheadings      = [] 


fheadings.append("Train on 90%, Test on first 10%")


rejectchars = [',','.','?','<','>','!','"','-','%','&','#','(',')','*',';'];
rcnt = len(rejectchars);


def remove_characters(word):
  """Replace special characters in the word"""
  
  for i in range(rcnt):
    rchar = rejectchars[i]
    if rchar in word:
      word = word.replace(rchar,"")

  return word


def remove_symbols(word):
  """Replace symbols in the word"""
  w = len(word)
  word = (ord(c) for c in word) 
  word = map(lambda x:x if x<123 or x>255 else " ", word)
  newword=""
  for c in range(w):
    if word[c] <> " ":
      newword += chr(word[c]);
  
  return newword


def find_word(word, sentence):
  """Find and return index of word in sentence"""

  masterlen = len(sentence)
  find=0
  temp="x"
  try:
   temp = sentence.index(word);
   return temp
  except ValueError:
   return temp


def find_phrase(phrase, syns):
  """Find and return index of Synset in sentence"""

  masterlen = len(syns)
  find=0
  temp="x"
  try:
   temp = syns.index(phrase);
   return temp
  except ValueError:
   return temp



def document_features(sentence, keyword):
  """ For a given word and sentence                                                    """
  """ The Lesk algorithm returns a Synset with the highest number of overlapping words """
  """ between the context sentence and different definitions from each Synset.         """

  features = {}
  synsent  = ""
  phrase   = lesk(sentence,keyword)
  findx = find_phrase(phrase, syns)
  if (findx != "x"):
    synsent = synsdef[findx]
    synsent = synsent.encode('ascii')
   
 
  features['(%s : %s)' % (sentence, phrase)]  = synsent
  
  return features

def document_features1(sentence, keyword):
  """ For a given word and sentence                                                    """
  """ The Lesk algorithm returns a Synset with the highest number of overlapping words """
  """ between the context sentence and different definitions from each Synset.         """

  features = {}
  synsent  = ""
  phrase   = lesk(sentence,keyword)
  findx = find_phrase(phrase, syns)
  s = ""
  results1 = ""
  if (findx != "x"):
    s = synsname[findx]
    synsent = synsdef[findx]
    synsent = synsent.encode('ascii')
    results1 = swn.senti_synset(s)
    temp = ("%s %s")  % (keyword,results1)
    features['(%s : %s)' % (keyword, phrase)]  = results1
    
  
  return features



print
print
print
defaultword = 'look'
keyword = raw_input("Please enter an ambiguous word in Alice in Wonderland ")
print ("or Press return to use the default word %s") % (defaultword)
valid = 0
if keyword == "":
     keyword =  defaultword

print
print


sentences = ""
sentword  = []
corpus     = "Alice in Wonderland"
fullcorpus = "Alice in Wonderland by Lewis Carroll"
cwd = os.getcwd()
currfilepath = str(cwd) + "\carroll-alice.txt"
print currfilepath
print ("Enter the Full File Path including the File")
print ("or Press return to use current File Path %s") % (currfilepath)
filepath = raw_input("Please enter the File Path now ")
valid = 0
if filepath == "":
     filepath = currfilepath

 
try:
       f = open(filepath,"r")
       try:
         valid=1
         x =0
         j=0
         findword =0
         for lines in f:
           lines = lines.rstrip()
           temp = lines.split(" ");
           l = len(temp)
           senttemp = ""
           for x in range(l):
             word = remove_characters(temp[x])
             word = remove_symbols(word)
             word = word.lower()
             word = word.replace(" ","")
             if (word != ''):
               if (word == keyword):
                 findword=1

               allwords.append(word)
               if senttemp == "":
                 senttemp = word
               else:
                 senttemp = senttemp + "," + word 
           
           if (findword == 1):
               sentword.append(senttemp) 

           findword = 0        
       finally:
            f.close()
         
except IOError:
       print ("File not Found - Program aborting")

if not(valid):
   exit();



# Trim sentence and use 2 words before and after key word
print
print("Trimming Sentences in %s") % (corpus)
indx = 0
sl = len(sentword)
for i in range(sl):
   l = len(sentword[i])
   s = sentword[i].split(",")
   #s = s.split(",")
   indx = s.index(keyword)
   start = 0
   fin   = l
   if ((indx > 1) and (indx < l)):
     start = indx - 2
     fin   = indx + 3
   else:
     if (indx <= 1):
       start = 0
     if (indx >= l-1):
       fin = l

   s1 = s[start:fin]
   sentwordadj.append(s1)

sla = len(sentwordadj)



# Store Synset defintions for keyword 
print
print
print("Synsets for Keyword: %s") % (keyword)
print
for ss in wn.synsets(keyword):
    syns.append(ss)
    synsdef.append(ss.definition())
    temp = ss.name()
    temp = temp.encode('ascii')
    synsname.append(temp)
    print("%s\t%s") % (ss,ss.definition())
  



 
# Get Sysnet features for each trimmed sentence
print
print
print("Getting feature set for each trimmed sentence")
featuresets=[]
for i in range(sla):
  featuresets.append(document_features(sentwordadj[i], keyword))




print("Corpus: %s  %i instances of KeyWord: %s") % (fullcorpus,sl,keyword)
print
print("Sentence\tSense Tag\tSense Tag Description")
print
for i in range(sla):
  print
  print featuresets[i]



print
print("Getting Sentiment Value for each trimmed sentence")
print
featuresets1=[]
fs = []
fs1 = []
fs2 = []
fs3 = []
for i in range(sla):
  phrase   = lesk(sentwordadj[i],keyword)
  findx    = find_phrase(phrase, syns)
  s = ""
  results = ""
  if (findx != "x"):
    s = synsname[findx]
    synsent = synsdef[findx]
    synsent = synsent.encode('ascii')
    results1 = swn.senti_synset(s)
    pscore = results1.pos_score()
    nscore = results1.neg_score()
    temp = ("%s\t%.2f\t%.2f")  % (syns[findx],pscore,nscore)
    featuresets1.append(temp)
   



l = len(featuresets1)
print
# Train on 90%, Test on 10%
testlim  = int(l *.01)
train_set, test_set  = featuresets[testlim:], featuresets[:testlim]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#accuracy   = nltk.classify.accuracy(classifier, test_set)
#results.append(accuracy)
#print(classifier.show_most_informative_features(5))

print
print
print
#l = len(results)
#print ("%s\t%s\t%s") % ("Feature" , "Feature Desc                              ", "                Accuracy")
#indx = 0
#for i in range(l):
  #indx = indx + 1
  #print("%d\t%s\t%.4f") % (indx, fheadings[i], results[i])



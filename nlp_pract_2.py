
from gettext import npgettext
from pyexpat import model
import gensim
from gensim import corpora
from gensim import models
import numpy as np 

text1 = ["""Gensim is a free open-source Python library for representing documents as semantic vectors,
           as efficiently and painlessly as possible. Gensim is designed 
           to process raw, unstructured digital texts using unsupervised machine learning algorithms."""]

tokens1 = [[item for item in line.split()] for line in text1]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

print("-----------------------------------------------------------------------------------")

from gensim.utils import simple_preprocess
from gensim import corpora

text2 = open('sample.txt', encoding ='utf-8')
 
tokens2 =[]
for line in text2.read().split('.'):
  tokens2.append(simple_preprocess(line, deacc = True))

g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " +str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)



print("-----------------------------------------------------------------------------------------")

g_dict1.add_documents(tokens2)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

print("-----------------------------------------BAG OF WORDS---------------------------------------------")


g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("Bag of Words : ", g_bow)


print("-----------------------------------------------------------------------------------------------")


# Save the Dictionary and BOW
corpora.MmCorpus.serialize('D:/g_bow1.mm', g_bow) 

# Load the Dictionary and BOW
#g_dict_load = corpora.Dictionary.load('/content/drive/MyDrive/gensim_tutorial/g_dict1.dict')
#g_bow_load = corpora.MmCorpus('D:/g_bow1.mm')


print("--------------------------------------------------TF- IDF----------------------------------------------------")


text = ["The food is excellent but the service can be better",
        "The food is always delicious and loved the service",
        "The food was mediocre and the service was terrible"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])
# Loading pretrained Word2Vec embeddings and using this to determine
# similarity between a predetermined set of sentences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim, torch, nltk, re
from gensim.test.utils import common_texts, datapath
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim import utils
from nltk.corpus import stopwords
from sklearn.manifold import TSNE

# Simple sentence embeddings using average of word embeddings in each sentence
def EmbedSentences(model, sent):
    sentEmbedding = [None] * len(sent)
    for i, e1 in enumerate(sent):
        currSent = [None] * len(e1.split())
        for j, e2 in enumerate(e1.split()):
            try: 
                currSent[j] = model[e2]
            except KeyError:
                currSent[j] = np.zeros(len(model["king"])) # replace unknown words by a zero vector
        sentEmbedding[i] = np.array(currSent).mean(axis=0)
    return(sentEmbedding)

# calculating cosine between two vectors
def cos_sim(vec1, vec2):
    return(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

# creating a matrix of size # of sent x # of sent with each cell corresponding to the cosine similarity between the given sentences
def cosMat_calc(sentEmbedding):
    list1 = ["s"] * len(sent)
    list2 = list(range(0, len(sent)))
    labels = [x + str(y) for x, y in zip(list1, list2)]
    cosMat = pd.DataFrame(np.zeros((len(labels), len(labels))))
    cosMat.index = labels
    cosMat.columns = labels
    for i, sent1 in enumerate(labels):
        for j, sent2 in enumerate(labels):
            cosMat.iloc[i,j] = cos_sim(sentEmbedding[int(labels[i][1:])], 
                                       sentEmbedding[int(labels[j][1:])])
    return(cosMat)

# Reading in sentences to be analyzed... 
sentDF = pd.read_table("C:\\Users\\putne\\OneDrive\\Desktop\\UNIGEFall2023\\NLP\\TP2\\T_sent.txt",
              names=["ID", "Sentence"])
sent = list(sentDF["Sentence"])

# Basic text processing, converting to lower, removing extraneous characters
print("Preprocessing text...")
replace_space = re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
stopwords = set(stopwords.words('english'))

for i,e in enumerate(sent):
    text = sent[i]
    text = text.lower() # lowercase text
    text = replace_space.sub(' ', text) # replace replace_space symbols by space in text
    text = bad_symbols.sub('', text) # delete symbols which are in bad_symbols from text
    text = ' '.join(word for word in text.split() if word not in stopwords) # delete stopwords from text
    sent[i] = text

## ------------ Sentence similarity using pretrained Word2Vec embeddings ------------------------ ##
# using embeddings pre-trained on 'Google News Dataset', which consists of about 100 billion words
# wv = api.load('word2vec-google-news-300', return_path=True)
# dataPath = "C:\\Users\\putne\\OneDrive\\Desktop\\UNIGEFall2023\\NLP\\TP2\\word2vec-google-news-300\\GoogleNews-vectors-negative300.bin"
dataPath = input("Please provide the path to the pre-trained Word2Vec embeddings: ")

print("Loading pre-trained vectors...")
model = gensim.models.KeyedVectors.load_word2vec_format(datapath(dataPath)
                                                         , binary=True)
# Simple sentence embeddings using average of word embeddings in each sentence
print("Averaging over word embeddings to create sentence embeddings...")
sentEmbed = EmbedSentences(model, sent)

# Calculating cos similarity between sentence vectors
print("Finding the most similar sentence based on cosine similarity...")
cosMat = cosMat_calc(sentEmbed)

# Finding the most similar sentence for each sentence
mostSimilarSentence = pd.DataFrame(round(cosMat, 3).replace(1, 0).idxmax()) # rounding because some 1s are not exactly 1s for numerical reasons
arr1 = [val for val in mostSimilarSentence.index]
arr2 = [val for val in mostSimilarSentence[0]]
mostSimilarSentence = pd.DataFrame({'sent1' : arr1, 'sent2' : arr2})

# Dictionary to convert sentences to IDs
list1 = ["s"] * len(sent)
list2 = list(range(0, len(sent)))
labels = [x + str(y) for x, y in zip(list1, list2)]
sentIdDict = dict(zip(labels, sentDF['ID']))

# Visualizing using TSNE
tsne = TSNE(n_components=2)
tsne_result = pd.DataFrame(tsne.fit_transform(pd.DataFrame(sentEmbed)))

plt.scatter(tsne_result[0], tsne_result[1])
for i in range(0, len(tsne_result)):
    plt.text(tsne_result.iloc[i, 0], tsne_result.iloc[i, 1],
             sentDF['ID'][i])
    
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("TSNE of Select Sentences Using Averaged Word2Vec Embeddings")
plt.show()

# Most similar sentence
mostSimilarSentence = mostSimilarSentence.replace(sentIdDict)
print(mostSimilarSentence)

# Determining the accuracy
acc = (mostSimilarSentence["sent1"] == mostSimilarSentence["sent2"]).mean()
print("The method yields an accuracy of", round(acc, 3), "for the given dataset.")

# Saving each sentence along with its most similar as 'out1.2.txt'
#with open('C:\\Users\\putne\\OneDrive\\Desktop\\UNIGEFall2023\\NLP\\TP2\\out1.2.txt', 'a') as f:
#    dfAsString = mostSimilarSentence.to_string(header=False, index=False)
#    f.write(dfAsString)
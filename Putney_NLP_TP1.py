import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import math

# creating cooccurence matrix with window size of 5 - dim T x B
# text input is a list of all words in the text in order having removed all extraneous characters
# i.e. the sentence structure is lost
def coMatrix_calc2(text, T, B):
    coMatrix = pd.DataFrame(np.zeros((len(T), len(B))))
    coMatrix.columns = B
    coMatrix.index = T
    for targ in T:
        match = np.where(text2 == targ)[0]
        for i in match:
            upper = i + 3
            lower = i - 2
            wind = text2[lower:upper]
            for word in B:
                if word in wind:
                    try:
                        coMatrix.loc[targ, word] += 1
                    except Exception:
                        pass
    return(coMatrix)

# converting counts from cooccurence matrix to PPMI scores
# i.e. PPMI(w,c) = max(log_2(p(w,c)/(p(w)p(c))), 0)
def PPMI_calc(coMatrix):
    PPMIMat = pd.DataFrame(np.zeros(coMatrix.shape))
    PPMIMat.columns = coMatrix.columns
    PPMIMat.index = coMatrix.index
    total = coMatrix.sum().sum() # total number of cooccurences
    for i,e in enumerate(coMatrix.columns):
        p_c = coMatrix[e].sum() / total 
        for j, f in enumerate(coMatrix.index):
            p_wc = coMatrix.loc[f, e] / total
            p_w = coMatrix.loc[f].sum() / total 
            try:
                PPMIMat.iloc[j,i] = max(0, math.log(p_wc / (p_w * p_c),2))
            except:
                pass
    return(PPMIMat)


# calculating cosine between two vectors
def cos_sim(vec1, vec2):
    return(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

# creating a matrix of size T x T with each cell corresponding to the cosine similarity between the given words
def cosMat_calc(T, PPMIMat):
    cosMat = pd.DataFrame(np.zeros((len(T), len(T))))
    cosMat.index = T
    cosMat.columns = T
    for i, word1 in enumerate(T):
        for j, word2 in enumerate(T):
            cosMat.iloc[i,j] = cos_sim(PPMIMat.iloc[i], PPMIMat.iloc[j])
    return(cosMat)

# creating a matrix of size T x 3 with the first two columns corresponding to a word and it's most similar word
# and the third column is the cosine similarity between those words
def closeWord_calc(T, cosMat):
    closestWord = pd.DataFrame(np.zeros((len(T), 3)))
    closestWord.columns = ["Word", "Closest", "Cos_Similarity"]
    for i, targ in enumerate(cosMat.index):
        redList = cosMat.iloc[i,:][cosMat.iloc[i,:].index != targ] # removing current word
        val = sorted(list(redList), reverse = True)[0] # getting highest value after removing current word
        word = redList.index[np.where(redList == val)][0]
        closestWord.iloc[i, 0] = targ
        closestWord.iloc[i, 1] = word
        closestWord.iloc[i, 2] = val
    return(closestWord)

# ----------------------- PREPROCESSING ----------------------------------- # 

# file_text = 'C:\\Users\\putne\\OneDrive\\Desktop\\UNIGEFall2023\\NLP\\TP1\\TEXT_LM_FULL.txt'
file_text = input("Path to the raw text to be analyzed: ") 
print("Processing text...")
text2 = ""
with open(file_text) as f:
    for line in f:
        text2 = text2 + line

# removing newline characters and extra spaces
text2 = text2.lower().replace('\n', ' ').replace("\\", "").replace("  ", " ")
text2 = text2.replace("?", " ").replace("!", " ").replace("m.", "m").replace(".", " ")
text2 = text2.replace(",", " ").replace(";", " ").replace("-", "")
text2 = np.array(text2.split(" "))
text2 = np.delete(text2, np.where(text2 == ''))

# ----------------------- INPUT ------------------------------------------ #
# B is a text file containing select words from 'Les Misérables', 
# in order, separated each by a newline.
# file_B = 'C:\\Users\\putne\\OneDrive\\Desktop\\UNIGEFall2023\\NLP\\TP1\\B.txt'
file_B = input("Path to 'B', a .txt file containing words to be used as features, separated by newlines: ")
B = ""
with open(file_B, 'r') as f:
    for line in f:
        B = B + line
B = B.split("\n")

# T is a text file containing target words for which we 
# will try to determine their similarity.
# file_T = 'C:\\Users\\putne\\OneDrive\\Desktop\\UNIGEFall2023\\NLP\\TP1\\T.txt'
file_T = input("Path to 'T', a .txt file containing target words of interest, separated by newlines: ")
T = ""
with open(file_T, 'r') as f:
    for line in f:
        T = T + line
T = T.split("\n")
T = T[:-1]

# ----------------------- CO-OCCURENCE MATRIX ---------------------------- #

# Creating coccurence counts matrix 
coMatrix = coMatrix_calc2(text2, T, B)
coMatrix+=0.1 # adding a small constant k to the count matrix as suggested at the end of 6.6 in Jurafsky 
#coMatrix = coMatrix.loc[:, coMatrix.sum(axis=0) > 0] # removing rows with no counts

# Transforming counts to PPMI
PPMIMat = PPMI_calc(coMatrix)

# ----------------------- PCA AND PLOTTING ------------------------------- #

dataset=PPMIMat
#PCA
pca=PCA()
pca.fit(dataset)
#Applying PCA 2 components:
pca=PCA(n_components=2)
pca.fit(dataset)
pca_data=pca.transform(dataset)

#If we have labels:
labels=dataset.index

####Plotting##
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
xs =  pca_data[:,0] # first component
ys =  pca_data[:,1] # second component

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Select Words in 'Les Misérables'")

ax.scatter(xs, ys, s=50, alpha=0.6, edgecolors='w')

xs =  pca_data[:,0] + [0,0.2,-0.6,0,0.2,0.3,0,0,-0.5,-0.4,-0.1,0.3,-0.3,0,-0.8,-0.3,0,-0.2,0.2]
ys =  pca_data[:,1] + [0,0.2,-0.1,0,0.2,0,0,0,0.2,-0.2,-0.2,0.2,-0.3,0.3,0.2,0.15,0,-0.2,0.2]

for x, y, label in zip(xs, ys, labels):
    ax.text(x, y, label)

plt.show()

# ----------------------- MOST SIMILAR WORDS ----------------------------------- #
print("\n Calculating most similar words...")

cosMat = cosMat_calc(T, PPMIMat)
#print(cosMat)

closeWord = closeWord_calc(T, cosMat) # a closest to b doesn not apply b is closest to a in the case of c being equally close
closeSet = set()
for i in range(1, len(closeWord)):
    closeSet.add((closeWord.iloc[i,0], closeWord.iloc[i,1], round(closeWord.iloc[i,2],3)))

print("\n Most similar word pairs along with their cosine similarity...")
print(closeSet)

print("\n Rearranging information in a table for easier viewing...")
print(closeWord)

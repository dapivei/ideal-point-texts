"""Preprocess UK Parlimentary speeches using methods described by Vafa et al.

### References
[1] Vafa, Keyon, Suresh Naidu, and David M. Blei. Text-Based Ideal Points, (2020). 
    https://github.com/keyonvafa/tbip/tree/master
"""

import os
import sys 

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 

utils_dir =  os.path.join(project_dir, "setup")
sys.path.append(utils_dir)

import setup_utils as utils

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

data_dir = os.path.join(project_dir, "data/parlspeech-2018-2019/raw")
save_dir = os.path.join(project_dir, "data/parlspeech-2018-2019/clean")

#%%

df = pd.read_csv(os.path.join(data_dir, 'Corp_HouseOfCommons_V2_2018-2019.csv'), 
                 encoding="ISO-8859-1")

#%%

# removing speeches given by the chair (speeches serve for debate organization only)
df = df[~df['chair']]

# removing speeches without recorded party affiliation
df = df[~df['party'].isnull()]

# removing speeches without recorded speaker
df = df[~df['speaker'].isnull()]

# ADDED: removing speeches about business of the house
df = df[df['agenda']!='Business of the House']

# removing speeches with fewer than 50 words
min_words = 50
df = df[df['terms'] > min_words]

# removing speakers with fewer than 24 speeches 
# (WHAT ABOUT PEOPLE WHO BELONG TO MORE THAN ONE PARTY?)
min_speeches = 24
num_speeches = df.groupby(['speaker','party'])['text'].count()
cutoff = num_speeches[num_speeches < min_speeches]
for name, party in cutoff.index:
  df = df[~((df['speaker']==name) & (df['party']==party))]

speaker = np.array(df['speaker'])
party = np.array(df['party'])
speeches = np.array(df['text'])

#%%

# Create mapping between names and IDs
speaker_party = np.array(
    [(speaker[i] + " (" + party[i] + ")").title() for i in range(len(speaker))])

speaker_to_speaker_id = dict(
    [(y.title(), x) for x, y in enumerate(sorted(set(speaker_party)))])
author_indices = np.array(
    [speaker_to_speaker_id[s.title()] for s in speaker_party])
author_map = np.array(list(speaker_to_speaker_id.keys()))

stopwords = []
with open(os.path.join(project_dir, "setup/stopwords/HouseOfCommons_stop.txt"), "r") as file:
  for line in file:
     line_strip = line.replace('\n', '')
     stopwords.append(str(line_strip))

count_vectorizer = CountVectorizer(min_df=0.001,
                                   max_df=0.3, 
                                   stop_words=stopwords, 
                                   ngram_range=(1, 3),
                                   token_pattern="[a-zA-Z]+")

#%%

# Learn initial document term matrix. This is only initial because we use it to
# identify words to exclude based on author counts.
counts = count_vectorizer.fit_transform(speeches)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                            key=lambda kv: kv[1])])

#%%

# Remove phrases spoken by less than 10 Senators.
counts_per_author = utils.bincount_2d(author_indices, counts.toarray())
min_authors_per_word = 10
author_counts_per_word = np.sum(counts_per_author > 0, axis=0)
acceptable_words = np.where(
    author_counts_per_word >= min_authors_per_word)[0]

#%% 

# Fit final document-term matrix with modified vocabulary.
count_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                   vocabulary=vocabulary[acceptable_words])
counts = count_vectorizer.fit_transform(speeches)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                            key=lambda kv: kv[1])])

#%%

# Adjust counts by removing unigram/n-gram pairs which co-occur.
counts_dense = utils.remove_cooccurring_ngrams(counts, vocabulary)

# Remove speeches with no words.
existing_speeches = np.where(np.sum(counts_dense, axis=1) > 0)[0]
counts_dense = counts_dense[existing_speeches]
author_indices = author_indices[existing_speeches]

#%%

# Save data.
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# `counts.npz` is a [num_documents, num_words] sparse matrix containing the
# word counts for each document.
sparse.save_npz(os.path.join(save_dir, "counts.npz"),
                sparse.csr_matrix(counts_dense).astype(np.float32))
# `author_indices.npy` is a [num_documents] vector where each entry is an
# integer indicating the author of the corresponding document.
np.save(os.path.join(save_dir, "author_indices.npy"), author_indices)
# `vocabulary.txt` is a [num_words] vector where each entry is a string
# denoting the corresponding word in the vocabulary.
np.savetxt(os.path.join(save_dir, "vocabulary.txt"), vocabulary, fmt="%s")
# `author_map.txt` is a [num_authors] vector of strings providing the name of
# each author in the corpus.
np.savetxt(os.path.join(save_dir, "author_map.txt"), author_map, fmt="%s")

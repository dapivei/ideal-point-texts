import os
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


class UKParliamentSpeechPreprocessor:
    def __init__(self, data_dir, save_dir, data_file, min_words, min_speeches):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.data_file = data_file
        self.min_words = min_words
        self.min_speeches = min_speeches
        self.stopwords_file = os.path.join(
            os.path.dirname(__file__), 
            "setup/stopwords/HouseOfCommons_stop.txt"
            )

    def _load_data(self):
        print("Loading data...")
        start_time = time.time()
        df = pd.read_csv(
            f"{self.data_dir}/{self.data_file}", 
            encoding="ISO-8859-1", 
            )
        load_time = time.time() - start_time
        print(f"Data loaded. Time taken: {load_time:.2f} seconds")
        return df

    def _preprocess_data(self, df):
        print("Preprocessing data...")
        start_time = time.time()
        print("Initial number of rows:", len(df))
        
        df = self._filter_data(df)
        
        print("Filtered number of rows:", len(df))
        
        speaker, party, speeches = df['speaker'].values, df['party'].values, df['text'].values
        preprocess_time = time.time() - start_time
        print(f"Data preprocessed. Time taken: {preprocess_time:.2f} seconds")
        return speaker, party, speeches

    def _filter_data(self, df):
        #TODO: parametrized period required in the filter

        print("Filtering data...")
        start_time = time.time()
        print("Initial number of rows:", len(df))
        
        df = df[~df['chair']]
        df = df.dropna(subset=['party', 'speaker'])
        df = df[df['terms'] > self.min_words]
        df = df[df['agenda']!='Business of the House']
        num_speeches = df.groupby(['speaker', 'party']).size()
        speakers_to_drop = num_speeches[num_speeches < self.min_speeches].index
        df = df[~df.set_index(['speaker', 'party']).index.isin(speakers_to_drop)]
        
        filter_time = time.time() - start_time
        print(f"Data filtered. Time taken: {filter_time:.2f} seconds")
        print("Filtered number of rows:", len(df))
        
        return df

    def preprocess(self):
        df = self._load_data()
        speaker, party, speeches = self._preprocess_data(df)
        return speaker, party, speeches

    def save_preprocessed_data(self, speaker, party, speeches):
        #TODO: resemble Simone's text, right now not doing exactly what it should
        print("Saving preprocessed data...")
        start_time = time.time()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Save data
        name = f"{self.min_words}wr_{self.min_speeches}sp"
        save_counts = f"{self.save_dir}/counts_{name}.npz"
        save_author_indices = f"{self.save_dir}/author_indices_{name}.npy"
        save_vocabulary = f"{self.save_dir}/vocabulary_{name}.txt"
        save_author_map = f"{self.save_dir}/author_map_{name}.txt"

        # Convert speeches to bag-of-words representation
        print("Converting speeches to bag-of-words representation...")
        count_vectorizer = CountVectorizer(
            stop_words='english', 
            token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
        counts = count_vectorizer.fit_transform(speeches)

        # Save preprocessed data
        print("Saving preprocessed data files...")
        sparse.save_npz(save_counts, counts.astype(np.float32))
        np.save(save_author_indices, speaker)
        np.savetxt(save_vocabulary, count_vectorizer.get_feature_names(), fmt="%s")
        np.savetxt(save_author_map, party, fmt="%s")
        
        save_time = time.time() - start_time
        print(f"Preprocessed data saved. Time taken: {save_time:.2f} seconds")
        print("Preprocessing complete.")

def preprocess_uk_parliament_speeches(
    data_dir, 
    save_dir, 
    data_file, 
    min_words, 
    min_speeches
    ):
    preprocessor = UKParliamentSpeechPreprocessor(
        data_dir, 
        save_dir, 
        data_file, 
        min_words, 
        min_speeches
        )
    speaker, party, speeches = preprocessor.preprocess()
    preprocessor.save_preprocessed_data(speaker, party, speeches)

# Run
data_dir = "/scratch/dp3766/text-base/ideal-point-texts/data/raw"
save_dir = "/scratch/dp3766/text-base/ideal-point-texts/data/prepro"
data_file = "Corp_HouseOfCommons_V2.csv"
min_words = 50
min_speeches = 24
start_time = time.time()
preprocess_uk_parliament_speeches(data_dir, save_dir, data_file, min_words, min_speeches)
total_time = time.time() - start_time
print(f"Total time taken for preprocessing: {total_time:.2f} seconds")

import os
import time
import numpy as np
import pandas as pd
import setup_utils as utils

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from functools import wraps


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


class ParliamentSpeechPreprocessor:
    def __init__(
        self, 
        base_dir,
        data_dir, 
        save_dir, 
        data_file, 
        min_words, 
        min_speeches, 
        min_df, 
        max_df,
        min_authors_per_word,
        topic_parts,
        start_date,
        end_date
        ):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.data_file = data_file
        self.min_words = min_words
        self.min_speeches = min_speeches
        self.min = min_df
        self.max = max_df
        self.stopwords = self._load_stopwords()
        self.min_authors_per_word = min_authors_per_word
        self.topic_parts = topic_parts
        self.start_date = start_date
        self.end_date = end_date


    def _load_data(self):
        print("Loading data...")
        start_time = time.time()
        df = pd.read_csv(
            f"{self.data_dir}/{self.data_file}", encoding="ISO-8859-1" 
            )
        load_time = time.time() - start_time
        print(f"Data loaded. Time taken: {load_time:.2f} seconds")
        return df
    
    
    def _load_stopwords(self):
        """
        Load stopwords from a file and return them as a list.
        """
        stopwords_path = os.path.join(self.base_dir, "setup/stopwords/HouseOfCommons_stop.txt")
        with open(stopwords_path, "r") as file:
            stopwords = [line.replace('\n', '') for line in file]
        return stopwords


    def _filter_data(self, df):
        #TODO: parametrized period required in the filter

        print("Filtering data...")
        start_time = time.time()
        print("Initial number of rows:", len(df))
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] > pd.to_datetime(self.start_date)) & (df['date'] < pd.to_datetime(self.end_date))]
        print(f"Number of rows after filtering by dates {self.start_date}-{self.end_date}:", len(df))
        df = df[~df['chair']]
        print(f"Number of rows after removing speeches corresponding to chair:", len(df))
        df = df.dropna(subset=['party', 'speaker'])
        print(f"Number of rows after removing rows without party or speaker:", len(df))
        df = df[df['terms'] > self.min_words]
        print(f"Number of rows after removing speeches not meeting minimum of words:", len(df))
        irrel_agendas = [
            'Business of the House', 
            'Summer Adjournment', 
            'May Adjournment', 
            'Easter Adjournment', 
            'Christmas Adjournment', 
            'Whitsun Adjournment',
            'Prorogation of Parliament', 
            "Prime Minister's Update"
            ]
        df = df[~df['agenda'].isin(irrel_agendas)]
        print(f"Number of rows after removing speeches with agenda of non-interest:", len(df))
        num_speeches = df.groupby(['speaker', 'party']).size()
        speakers_to_drop = num_speeches[num_speeches < self.min_speeches].index
        df = df[~df.set_index(['speaker', 'party']).index.isin(speakers_to_drop)]
        party_counts = df.groupby(['agenda'])['party'].nunique()
        large_agendas = party_counts[party_counts > self.topic_parts].index.tolist()
        df = df[df['agenda'].isin(large_agendas)]
        filter_time = time.time() - start_time
        print(f"Data filtered. Time taken: {filter_time:.2f} seconds")
        
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
    
    def preprocess(self):
        df = self._load_data()
        speaker, party, speeches = self._preprocess_data(df)
        return speaker, party, speeches


    @timing_decorator
    def create_speaker_mapping(self, speaker, party):
        # Create speaker_party array
        print("Create speaker_mapping ...")
        speaker_party = np.array([(s + " (" + p + ")").title() for s, p in zip(speaker, party)])

        # Create speaker to speaker_id mapping
        unique_speaker_party = sorted(set(speaker_party))
        speaker_to_speaker_id = {s.title(): idx for idx, s in enumerate(unique_speaker_party)}

        # Create author_indices array
        author_indices = np.array([speaker_to_speaker_id[s.title()] for s in speaker_party])

        # Create author_map array
        author_map = np.array(list(speaker_to_speaker_id.keys()))

        return speaker_to_speaker_id, author_indices, author_map
    
    
    @timing_decorator
    def initialize_count_vectorizer(self):
        """
        Initialize CountVectorizer with specified parameters
        """
        print("Initialize count vectorizer...")

        coun_vect = CountVectorizer(
            min_df=self.min,
            max_df=self.max,
            stop_words=self.stopwords,
            ngram_range=(1, 3),
            token_pattern="[a-zA-Z]+"
            )
        return coun_vect


    @timing_decorator
    def learn_initial_dtm(self, speeches, coun_vect):
        """
        Learn initial document-term matrix
        """
        print("Learn initial document-term matrix...")
        counts = coun_vect.fit_transform(speeches)
        vocabulary = np.array(
            [k for (k, v) in sorted(coun_vect.vocabulary_.items(), key=lambda kv: kv[1])]
            )
        return counts, vocabulary


    @timing_decorator
    def filter_words_by_author_counts(self, counts, author_indices):
        """
        Filter words based on the number of authors who used them
        """
        print("Filter words based on the number of authors who used them...")
        counts_per_author = np.bincount(author_indices, minlength=counts.shape[1])
        author_counts_per_word = np.sum(counts > 0, axis=0)
        accept_words = np.where(author_counts_per_word >= self.min_authors_per_word)[0]
        return accept_words


    @timing_decorator
    def fit_final_dtm(self, speeches, vocabulary, accept_words):
        """
        Fit final document-term matrix with modified vocabulary.
        """
        print("Fit final document-term matrix with modified vocabulary...")

        c_v = CountVectorizer(
            ngram_range=(1, 3),
            vocabulary=vocabulary[accept_words]
            )
        counts = c_v.fit_transform(speeches)
        vocab = np.array(
            [k for (k, v) in sorted(c_v.vocabulary_.items(),key=lambda kv: kv[1])])
        return counts, vocab


    @timing_decorator
    def adjust_counts_and_filter_speeches(self, counts, vocabulary, author_indices):
        """
        Adjust counts by removing unigram/n-gram pairs which co-occur and filter speeches with no words.
        """
        print("Adjust counts by removing unigram/n-gram pairs which co-occur")

        counts_dense = utils.remove_cooccurring_ngrams(counts, vocabulary)
        
        # Remove speeches with no words
        existing_speeches = np.where(np.sum(counts_dense, axis=1) > 0)[0]
        counts_dense = counts_dense[existing_speeches]
        author_indices = author_indices[existing_speeches]

        return counts_dense, author_indices


    def save_preprocessed_data(self, counts_dense, author_indices, vocabulary, author_map):

        print("Saving preprocessed data...")
        start_time = time.time()

        # Name data
        name = f"{self.min_words}wr_{self.min_speeches}sp_{self.topic_parts}tp_{self.start_date}st_{self.end_date}nd"
        if not os.path.exists(f"{self.save_dir}/{name}"):
            os.makedirs(f"{self.save_dir}/{name}")       
        save_counts = f"{self.save_dir}/{name}/counts.npz"
        save_author_indices = f"{self.save_dir}/{name}/author_indices.npy"
        save_vocabulary = f"{self.save_dir}/{name}/vocabulary.txt"
        save_author_map = f"{self.save_dir}/{name}/author_map.txt"

        # Save preprocessed data
        print("Saving preprocessed data files...")
        sparse.save_npz(save_counts, sparse.csr_matrix(counts_dense).astype(np.float32))
        np.save(save_author_indices, author_indices)
        np.savetxt(save_vocabulary, vocabulary, fmt="%s")
        np.savetxt(save_author_map, author_map, fmt="%s")
        
        save_time = time.time() - start_time
        print(f"Preprocessed data saved. Time taken: {save_time:.2f} seconds")
        print("Preprocessing complete.")

def preprocess_parliament_speeches(
    base_dir,
    data_dir, 
    save_dir, 
    data_file, 
    min_words, 
    min_speeches,
    min_df,
    max_df,
    min_authors_per_word,
    topic_parts,
    start_date,
    end_date
    ):
    preprocessor = ParliamentSpeechPreprocessor(
        base_dir,
        data_dir, 
        save_dir, 
        data_file, 
        min_words, 
        min_speeches,
        min_df,
        max_df,
        min_authors_per_word,
        topic_parts,
        start_date,
        end_date
        )
    speaker, party, speeches = preprocessor.preprocess()
    speaker_to_speaker_id, author_indices, author_map = preprocessor.create_speaker_mapping(
        speaker, party
        )
    count_vect = preprocessor.initialize_count_vectorizer()
    counts, vocab = preprocessor.learn_initial_dtm(speeches, count_vect)
    accept_words = preprocessor.filter_words_by_author_counts(counts.toarray(), author_indices)
    counts, vocab = preprocessor.fit_final_dtm(speeches, vocab, accept_words)
    counts_dense, author_indices = preprocessor.adjust_counts_and_filter_speeches(counts, vocab, author_indices)
    preprocessor.save_preprocessed_data(counts_dense, author_indices, vocab, author_map)
    

# Run
base_dir = "/scratch/dp3766/text-base/ideal-point-texts/"
data_dir = "/scratch/dp3766/text-base/ideal-point-texts/data/raw"
save_dir = "/scratch/dp3766/text-base/ideal-point-texts/data/prepro"
data_file = "Corp_HouseOfCommons_V2.csv"
min_words = 50
min_speeches = 24
min_df = 0.001
max_df = 0.3
min_authors_per_word = 10
topic_parts = 2
start_date = '2018-01-01'
end_date = '2023-12-30'

start_time = time.time()
preprocess_parliament_speeches(
    base_dir,
    data_dir, 
    save_dir, 
    data_file, 
    min_words, 
    min_speeches, 
    min_df, 
    max_df, 
    min_authors_per_word,
    topic_parts,
    start_date,
    end_date
    )
total_time = time.time() - start_time
print(f"Total time taken for preprocessing: {total_time:.2f} seconds")
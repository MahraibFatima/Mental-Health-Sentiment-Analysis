from src.MentalHealthAnalysis import logger
from src.MentalHealthAnalysis.entity.config_entity import DataTransformationConfig
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')

class DataTransformation:
    def __init__(self, config):
        self.config = config
    
    def read_data(self):
        """Reads the dataset from the specified path."""
        self.data = pd.read_csv(self.config.data_path)
    
    def preprocessing(self):
        """Performs preprocessing steps such as handling missing values and text cleaning."""
        self.data.dropna(inplace=True)
        
        # Label encoding target variable 'y'
        lbl_enc = LabelEncoder()
        self.data['status'] = lbl_enc.fit_transform(self.data['status'].values)
        
        # Create additional features based on text length and number of sentences
        self.data['num_of_characters'] = self.data['statement'].str.len()
        self.data['num_of_sentences'] = self.data['statement'].apply(lambda x: len(sent_tokenize(x)))
        
        # Lowercasing the text
        self.data['statement'] = self.data['statement'].str.lower()
    
    def remove_patterns(self, text):
        """Removes unwanted patterns such as URLs, markdown links, handles, and punctuation."""
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def clean_text(self):
        """Applies the `remove_patterns` function to clean the text."""
        self.data['statement'] = self.data['statement'].apply(self.remove_patterns)
    
    def tokenize_and_stem(self):
        """Tokenizes and stems the statements."""
        self.data['tokens'] = self.data['statement'].apply(word_tokenize)
        
        # Initialize the stemmer
        stemmer = PorterStemmer()

        # Function to stem tokens
        def stem_tokens(tokens):
            return ' '.join(stemmer.stem(str(token)) for token in tokens)
        
        self.data['tokens_stemmed'] = self.data['tokens'].apply(stem_tokens)

    def vectorize_text(self):
        """Vectorizes the text using TF-IDF and combines it with numerical features."""
        # Split data into train and test sets
        X = self.data.drop('status', axis=1)
        y = self.data['status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
        
        # Fit and transform TF-IDF on training data
        X_train_tfidf = vectorizer.fit_transform(X_train['tokens_stemmed'])
        X_test_tfidf = vectorizer.transform(X_test['tokens_stemmed'])

        # Extract numerical features
        X_train_num = X_train[['num_of_characters', 'num_of_sentences']].values
        X_test_num = X_test[['num_of_characters', 'num_of_sentences']].values

        # Combine TF-IDF and numerical features
        X_train_combined = hstack([X_train_tfidf, X_train_num])
        X_test_combined = hstack([X_test_tfidf, X_test_num])

        print(f'Number of feature words: {len(vectorizer.get_feature_names_out())}')

        # Apply Random Over-Sampling on the vectorized data
        ros = RandomOverSampler(random_state=101)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_combined, y_train)

        print(f"Resampled training data shape: {X_train_resampled.shape}")
        return X_train_resampled, X_test_combined, y_train_resampled, y_test

    def train_test_split_save(self):
        """Splits the data into train/test sets and saves them to the specified directory."""
        train, test = train_test_split(self.data, test_size=0.2, random_state=101)
        #train_test_split(self.data, test_size=0.25, random_state=42)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        print(f"Training data shape: {train.shape}")
        print(f"Testing data shape: {test.shape}")


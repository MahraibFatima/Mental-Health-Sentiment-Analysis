import joblib
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to load the model
def load_model():
    try:
        # Load the model from the .joblib file
        model = joblib.load('artifacts/model_trainer/model.joblib')

        # Save the model to a .pkl file for future use
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Function to stem tokens
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(str(token)) for token in tokens)

# Function to preprocess the statement
def preprocess_statement(statement):
    # Example preprocessing logic
    statement = statement.lower().strip()
    tokens = word_tokenize(statement)  # Apply tokenization directly

    tokens_stemmed = stem_tokens(tokens)  # Apply stemming
    vectorizer = TfidfVectorizer()

    processed_statement = vectorizer.fit_transform([tokens_stemmed]).toarray()
    return processed_statement[0]  # Return the processed statement

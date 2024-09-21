from src.MentalHealthAnalysis import logger
from src.MentalHealthAnalysis.entity.config_entity import ModelTrainerConfig
import os
import pandas as pd
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.utils import resample  # If resampling is needed
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load train and test data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column].values.ravel()
        test_y = test_data[self.config.target_column].values.ravel()

        train_x['statement'] = train_x['statement'].fillna('').astype(str)
        test_x['statement'] = test_x['statement'].fillna('').astype(str) # Replace NaN with an empty string

        text_columns = ['statement', 'num_of_characters', 'num_of_sentences', 'tokens','tokens_stemmed']
        numerical_columns = train_x.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Apply TF-IDF to each text column separately using pipelines
        text_transformers = [(f'tfidf_{col}', Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000))
        ]), col) for col in text_columns]

        # Create a column transformer to apply TF-IDF to text columns and scaling to numerical columns
        # Define preprocessors for text and numeric data
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(), 'statement'),  # Use correct column name
                ('num', StandardScaler(), train_x.select_dtypes(include=['float64', 'int64']).columns.tolist())  # Adjust for your numeric column(s)
            ]
        )

        # Create a pipeline that first transforms the data, then fits the model
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(learning_rate=0.5, max_depth=7, n_estimators=500, random_state=101))
        ])

        # Train the model using the pipeline
        model_pipeline.fit(train_x, train_y)

        # Predict the labels on the test data
        y_pred = model_pipeline.predict(test_x)

        # Evaluate the model performance using accuracy
        accuracy = accuracy_score(test_y, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Ensure the root directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Save the trained model
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(model_pipeline, model_path)

        # Optionally print or log success
        print(f"Model saved at {model_path}")

    # Optional: Add a method for resampling
    def _resample(self, X, y):
        # Example of resampling
        df = pd.concat([pd.DataFrame(X), pd.Series(y, name='target')], axis=1)
        majority_class = df['target'].mode()[0]
        minority_class = df['target'].value_counts().index[1]
        
        df_majority = df[df['target'] == majority_class]
        df_minority = df[df['target'] == minority_class]
        
        df_minority_upsampled = resample(df_minority, 
                                         replace=True, 
                                         n_samples=len(df_majority), 
                                         random_state=101)
        
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        X_upsampled = df_upsampled.drop(['target'], axis=1)
        y_upsampled = df_upsampled['target']
        
        return X_upsampled, y_upsampled

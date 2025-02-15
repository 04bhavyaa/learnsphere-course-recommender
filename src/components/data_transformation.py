# src/components/data_transformation.py
import os
import sys
import spacy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    vectorizer_path: str = os.path.join('artifacts', 'vectorizer.pkl')
    processed_data_path: str = os.path.join('artifacts', 'processed_data.csv')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.nlp = spacy.load("en_core_web_sm")
        self.label_encoders = {}

    def clean_text(self, text):
        """Clean and normalize text data"""
        try:
            if pd.isna(text) or not isinstance(text, str):
                return ""
                
            # Convert to lowercase and process with spaCy
            doc = self.nlp(text.lower())
            
            # Remove stopwords, punctuation, and get lemmas
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop 
                     and not token.is_punct
                     and token.lemma_.strip()]
            
            return " ".join(tokens)
            
        except Exception as e:
            logging.error(f"Error in text cleaning: {str(e)}")
            raise CustomException(e, sys)

    def extract_features(self, text):
        """Extract additional features from course titles"""
        try:
            doc = self.nlp(text.lower())
            
            features = {
                'word_count': len([token for token in doc if not token.is_punct]),
                'tech_terms': len([token for token in doc if token.text in ['python', 'java', 'javascript', 'web', 'data', 'ai', 'ml']]),
                'has_numbers': any(token.like_num for token in doc),
                'avg_word_length': np.mean([len(token.text) for token in doc if not token.is_punct]) if doc else 0
            }
            
            return pd.Series(features)
            
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            raise CustomException(e, sys)

    def transform_categorical_features(self, df):
        """Transform categorical variables using label encoding"""
        try:
            categorical_cols = ['level', 'subject']
            
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            
            return df
            
        except Exception as e:
            logging.error(f"Error in categorical transformation: {str(e)}")
            raise CustomException(e, sys)

    def transform_numerical_features(self, df):
        """Scale numerical features"""
        try:
            numerical_cols = ['num_subscribers', 'num_reviews', 'content_duration']
            scaler = StandardScaler()
            
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            save_object(self.config.preprocessor_path, scaler)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in numerical transformation: {str(e)}")
            raise CustomException(e, sys)

    def create_text_vectors(self, df):
        """Create TF-IDF vectors from course titles"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Create and save text vectors
            text_vectors = vectorizer.fit_transform(df['processed_title'])
            vector_df = pd.DataFrame(
                text_vectors.toarray(),
                columns=[f'tfidf_{i}' for i in range(text_vectors.shape[1])]
            )
            
            # Save vectorizer
            save_object(self.config.vectorizer_path, vectorizer)
            
            return pd.concat([df, vector_df], axis=1)
            
        except Exception as e:
            logging.error(f"Error in text vectorization: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Main method to execute data transformation process"""
        try:
            logging.info("Started data transformation")
            
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Clean text data
            train_df['processed_title'] = train_df['course_title'].apply(self.clean_text)
            test_df['processed_title'] = test_df['course_title'].apply(self.clean_text)
            
            # Extract additional features
            train_features = train_df['processed_title'].apply(self.extract_features)
            test_features = test_df['processed_title'].apply(self.extract_features)
            
            train_df = pd.concat([train_df, train_features], axis=1)
            test_df = pd.concat([test_df, test_features], axis=1)
            
            # Transform categorical and numerical features
            train_df = self.transform_categorical_features(train_df)
            test_df = self.transform_categorical_features(test_df)
            
            train_df = self.transform_numerical_features(train_df)
            test_df = self.transform_numerical_features(test_df)
            
            # Create text vectors
            train_df = self.create_text_vectors(train_df)
            test_df = self.create_text_vectors(test_df)
            
            # Save processed data
            train_df.to_csv(self.config.processed_data_path, index=False)
            
            logging.info("Data transformation completed successfully")
            
            return (
                train_df,
                test_df,
                self.config.preprocessor_path,
                self.config.vectorizer_path
            )
            
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)
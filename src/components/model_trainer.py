# src/components/model_trainer.py
import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    vectorizer_file_path: str = os.path.join("artifacts", "vectorizer.pkl")
    model_metrics_path: str = os.path.join("artifacts", "model_metrics.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_recommendations(self, 
                               similarity_matrix: np.ndarray, 
                               test_indices: np.ndarray,
                               n_recommendations: int = 10) -> dict:
        """
        Evaluate recommendation quality using various metrics
        """
        try:
            # Calculate metrics
            coverage = np.mean(similarity_matrix > 0) * 100
            
            # Average similarity score for top N recommendations
            top_n_scores = np.sort(similarity_matrix, axis=1)[:, -n_recommendations:]
            avg_similarity = np.mean(top_n_scores)
            
            # Diversity score (1 - average similarity between recommendations)
            diversity = 1 - np.mean([
                np.mean(cosine_similarity(top_n_scores[i].reshape(1, -1), 
                                        top_n_scores[i:]))
                for i in range(len(top_n_scores)-1)
            ])
            
            metrics = {
                "coverage_percentage": float(coverage),
                "average_similarity_score": float(avg_similarity),
                "diversity_score": float(diversity),
                "number_of_courses": len(similarity_matrix)
            }
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_similarity_matrix(self, 
                               processed_data: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Create TF-IDF vectors and similarity matrix
        """
        try:
            # Initialize and fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Create TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(processed_data['processed_title'])
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix, vectorizer
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_training(self, data_path: str) -> dict:
        try:
            logging.info("Started model training")
            
            # Read processed data
            processed_data = pd.read_csv(data_path)
            logging.info(f"Loaded processed data with {len(processed_data)} courses")
            
            # Create similarity matrix and vectorizer
            similarity_matrix, vectorizer = self.create_similarity_matrix(processed_data)
            logging.info("Created similarity matrix and vectorizer")
            
            # Evaluate model
            test_indices = np.random.choice(
                len(processed_data), 
                size=min(1000, len(processed_data)), 
                replace=False
            )
            metrics = self.evaluate_recommendations(
                similarity_matrix, 
                test_indices
            )
            logging.info(f"Model evaluation metrics: {metrics}")
            
            # Save model artifacts
            save_object(
                file_path=self.model_trainer_config.vectorizer_file_path,
                obj=vectorizer
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    'similarity_matrix': similarity_matrix,
                    'course_ids': processed_data['course_id'].values
                }
            )
            
            # Save metrics
            pd.DataFrame([metrics]).to_json(
                self.model_trainer_config.model_metrics_path
            )
            
            logging.info("Model training completed successfully")
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)


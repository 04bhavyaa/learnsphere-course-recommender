# src/pipeline/prediction_pipeline.py
import os
import sys
import spacy
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, extract_keywords
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class PredictionConfig:
    model_path: str = os.path.join('artifacts', 'vectorizer.pkl')
    data_path: str = os.path.join('artifacts', 'processed_courses.csv')
    nlp_model: str = "en_core_web_sm"
    similarity_threshold: float = 0.1

@dataclass
class UserProfile:
    education: str
    skills: str
    career_goals: str
    learning_objectives: str
    
    def to_string(self) -> str:
        return f"{self.education} {self.skills} {self.career_goals} {self.learning_objectives}"
    
    def validate(self) -> tuple[bool, str]:
        if not all([self.education, self.skills, self.career_goals, self.learning_objectives]):
            return False, "All fields are required"
        if len(self.to_string()) < 10:
            return False, "Please provide more detailed information"
        return True, ""

class CourseRecommendation:
    def __init__(self, course_id: int, title: str, similarity: float, url: str = None, 
                 level: str = None, duration: float = None):
        self.course_id = course_id
        self.title = title
        self.similarity = similarity
        self.url = url
        self.level = level
        self.duration = duration
        
    def to_dict(self) -> dict:
        return {
            'course_id': self.course_id,
            'title': self.title,
            'similarity': round(self.similarity * 100, 2),
            'url': self.url,
            'level': self.level,
            'duration': self.duration
        }

class PredictionPipeline:
    def __init__(self):
        self.config = PredictionConfig()
        try:
            self.nlp = spacy.load(self.config.nlp_model)
            self.vectorizer = load_object(self.config.model_path)
            self.course_data = pd.read_csv(self.config.data_path)
            logging.info("Prediction pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing prediction pipeline: {str(e)}")
            raise CustomException(e, sys)

    def _validate_data(self) -> bool:
        """Validate if all required data is loaded properly"""
        if self.vectorizer is None:
            raise CustomException("Vectorizer not loaded", sys)
        if self.course_data is None or len(self.course_data) == 0:
            raise CustomException("Course data not loaded", sys)
        return True

    def _calculate_similarities(self, user_vector) -> np.ndarray:
        """Calculate similarity scores between user profile and courses"""
        try:
            course_vectors = self.vectorizer.transform(self.course_data['processed_title'])
            similarity_scores = cosine_similarity(user_vector, course_vectors).flatten()
            return similarity_scores
        except Exception as e:
            raise CustomException(f"Error calculating similarities: {str(e)}", sys)

    def _filter_recommendations(self, similarity_scores: np.ndarray, top_n: int) -> list[CourseRecommendation]:
        """Filter and format recommendations"""
        try:
            # Get indices of top recommendations above threshold
            valid_scores = similarity_scores >= self.config.similarity_threshold
            top_indices = similarity_scores.argsort()[-top_n:][::-1]
            top_indices = [idx for idx in top_indices if valid_scores[idx]]

            recommendations = []
            for idx in top_indices:
                course = self.course_data.iloc[idx]
                rec = CourseRecommendation(
                    course_id=int(course['course_id']),
                    title=course['course_title'],
                    similarity=similarity_scores[idx],
                    url=course.get('url', None),
                    level=course.get('level', None),
                    duration=course.get('content_duration', None)
                )
                recommendations.append(rec)

            return recommendations
        except Exception as e:
            raise CustomException(f"Error filtering recommendations: {str(e)}", sys)

    def predict(self, user_profile: UserProfile, top_n: int = 5) -> dict:
        """
        Generate course recommendations based on user profile
        
        Args:
            user_profile (UserProfile): User profile information
            top_n (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary containing recommendations and metadata
        """
        try:
            # Validate inputs
            self._validate_data()
            is_valid, error_msg = user_profile.validate()
            if not is_valid:
                return {
                    'status': 'error',
                    'message': error_msg,
                    'recommendations': []
                }

            # Process user profile
            user_keywords = extract_keywords(user_profile.to_string(), self.nlp)
            user_vector = self.vectorizer.transform([user_keywords])

            # Generate recommendations
            similarity_scores = self._calculate_similarities(user_vector)
            recommendations = self._filter_recommendations(similarity_scores, top_n)

            if not recommendations:
                return {
                    'status': 'warning',
                    'message': 'No relevant courses found. Try adjusting your profile information.',
                    'recommendations': []
                }

            return {
                'status': 'success',
                'message': f'Found {len(recommendations)} relevant courses',
                'recommendations': [rec.to_dict() for rec in recommendations]
            }

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

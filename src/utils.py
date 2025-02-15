# src/utils.py
import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def extract_keywords(text, nlp):
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
    return " ".join(keywords)
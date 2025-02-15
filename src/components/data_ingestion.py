# src/components/data_ingestion.py
import os
import sys
import pandas as pd
import mysql.connector
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "Neha@6283",
            "database": "learnsphere"
        }

    def connect_to_database(self):
        """Establish connection to MySQL database"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            logging.info("Successfully connected to MySQL database")
            return connection
        except Exception as e:
            logging.error(f"Error connecting to database: {str(e)}")
            raise CustomException(e, sys)

    def fetch_course_data(self, connection):
        """Fetch course data from database"""
        try:
            query = """
            SELECT 
                course_id,
                course_title,
                url,
                num_subscribers,
                num_reviews,
                level,
                content_duration,
                published_timestamp,
                subject
            FROM courses
            """
            df = pd.read_sql(query, connection)
            logging.info(f"Successfully fetched {len(df)} records from database")
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            raise CustomException(e, sys)

    def preprocess_raw_data(self, df):
        """Initial preprocessing of raw data"""
        try:
            # Convert timestamps
            df['published_timestamp'] = pd.to_datetime(df['published_timestamp'])
            
            # Handle missing values
            df['course_title'] = df['course_title'].fillna('')
            df['level'] = df['level'].fillna('Not Specified')
            df['subject'] = df['subject'].fillna('Other')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['course_id'])
            
            # Basic data validation
            df = df[df['content_duration'] >= 0]
            df = df[df['num_subscribers'] >= 0]
            df = df[df['num_reviews'] >= 0]
            
            logging.info("Initial data preprocessing completed")
            return df
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """Main method to execute data ingestion process"""
        try:
            logging.info("Started data ingestion")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Connect to database and fetch data
            connection = self.connect_to_database()
            df = self.fetch_course_data(connection)
            connection.close()
            
            # Preprocess the data
            df = self.preprocess_raw_data(df)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")
            
            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)
# src/pipeline/training_pipeline.py
import os
import sys
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join(os.getcwd(), "artifacts")

class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        os.makedirs(self.training_pipeline_config.artifacts_dir, exist_ok=True)
        
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
        self.pipeline_metrics = {
            "data_ingestion": {},
            "data_transformation": {},
            "model_training": {}
        }

    def validate_pipeline_results(self):
        """
        Validate pipeline results and artifacts
        """
        try:
            required_files = [
                "raw_courses.csv",
                "processed_courses.csv",
                "vectorizer.pkl",
                "model.pkl",
                "model_metrics.json"
            ]
            
            for file in required_files:
                file_path = os.path.join(self.training_pipeline_config.artifacts_dir, file)
                if not os.path.exists(file_path):
                    raise Exception(f"Required artifact not found: {file}")
                    
            # Validate data sizes
            raw_data = pd.read_csv(os.path.join(
                self.training_pipeline_config.artifacts_dir, 
                "raw_courses.csv"
            ))
            processed_data = pd.read_csv(os.path.join(
                self.training_pipeline_config.artifacts_dir, 
                "processed_courses.csv"
            ))
            
            if len(raw_data) != len(processed_data):
                raise Exception("Data size mismatch between raw and processed data")
                
            logging.info("Pipeline validation completed successfully")
            return True
            
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> dict:
        """
        Execute the complete training pipeline
        """
        try:
            logging.info("Starting training pipeline")
            
            # Data Ingestion
            logging.info("Starting data ingestion")
            train_data_path = self.data_ingestion.initiate_data_ingestion()
            self.pipeline_metrics["data_ingestion"] = {
                "data_path": train_data_path,
                "num_records": len(pd.read_csv(train_data_path))
            }
            logging.info("Data ingestion completed")
            
            # Data Transformation
            logging.info("Starting data transformation")
            transformed_data_path = self.data_transformation.initiate_data_transformation(
                train_data_path
            )
            self.pipeline_metrics["data_transformation"] = {
                "data_path": transformed_data_path,
                "num_records": len(pd.read_csv(transformed_data_path))
            }
            logging.info("Data transformation completed")
            
            # Model Training
            logging.info("Starting model training")
            model_metrics = self.model_trainer.initiate_model_training(
                transformed_data_path
            )
            self.pipeline_metrics["model_training"] = model_metrics
            logging.info("Model training completed")
            
            # Validate Pipeline Results
            self.validate_pipeline_results()
            
            logging.info("Training pipeline completed successfully")
            return self.pipeline_metrics
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    metrics = pipeline.run_pipeline()
    print("Pipeline Metrics:", metrics)
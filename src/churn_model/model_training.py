import pandas as pd
from .churn_predictor import ChurnPredictor
from src.logger_config import load_logger
import argparse
import sys
import os
import json
from datetime import datetime

logger = load_logger('ModelTraining')

def train_and_save_model(input_file, output_model_dir, output_report_dir):
    """Train the ChurnPredictor model on the input data, save the model and report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  
    logger.info("Starting model training process")
    
    try:
        os.makedirs(output_model_dir, exist_ok=True)
        os.makedirs(output_report_dir, exist_ok=True)
        logger.info(f"Ensured directories exist: {output_model_dir}, {output_report_dir}")
        
        # Load the training data
        logger.info(f"Loading training data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        predictor = ChurnPredictor()
        logger.info(f"Initialized ChurnPredictor with model: {predictor.model_name}")
        
        logger.info("Starting model training")
        training_report = predictor.train(df)
        
        # Save the model
        model_filename = os.path.join(output_model_dir, f'churn_model_{timestamp}.pkl')
        logger.info(f"Saving model to {model_filename}")
        predictor.save_model(model_filename)
        
        # Save the training report
        report_filename = os.path.join(output_report_dir, f'training_report_{timestamp}.json')
        logger.info(f"Saving training report to {report_filename}")
        with open(report_filename, 'w') as f:
            json.dump(training_report, f, indent=4)
        
        logger.info(f"Model training completed successfully. Model: {predictor.model_name}")
        logger.info(f"Model saved to {model_filename}")
        logger.info(f"Training report saved to {report_filename}")
        logger.info(f"Training report: {training_report}")
        
 
        return training_report
        
    except FileNotFoundError:
        logger.error(f"Input file {input_file} not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a churn prediction model.")
    parser.add_argument('--input', type=str, default='datasets/Customer-Churn.csv', help='Path to the input training CSV file')
    parser.add_argument('--model-dir', type=str, default='models/', help='Directory to save the trained model')
    parser.add_argument('--report-dir', type=str, default='reports/', help='Directory to save the training report')
    args = parser.parse_args()
    
    train_and_save_model(args.input, args.model_dir, args.report_dir)
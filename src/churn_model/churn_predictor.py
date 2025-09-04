import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
from datetime import datetime
from src.logger_config import load_logger


# Initialize logger at module level
logger = load_logger('ChurnPredictor')

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.model_name = "RandomForest"
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_columns = []
        self.categorical_columns = []
        self.training_report = {}
        self.numerical_imputation_values = {}
        self.categorical_imputation_values = {}
        

    def validate_data(self, df, is_training=True):
        """Validate input data for required columns and data types, and strip spaces from column names"""
        logger.info("Starting data validation")
        # Strip spaces from column names
        df.columns = df.columns.str.strip().str.lower()
        required_columns = [
            'gender', 'senior_citizen', 'is_married',
            'dependents', 'tenure', 'internet_service', 'contract',
            'payment_method', 'monthly_charges', 'total_charges',
            'phone_service', 'dual', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv',
            'streaming_movies', 'paperless_billing'
        ]
        if is_training:
            required_columns.append('churn')
        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
        for col in numerical_columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                try:
                    logger.info(f"Converting column {col} to numeric")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logger.error(f"Invalid data type in column {col}")
                    raise ValueError(f"Invalid data type in column {col}")
        
        logger.info("Data validation completed")
        return df


    def clean_data(self, df, is_training=True):
        """Clean the dataset by handling missing values, using stored values for inference"""
        logger.info("Starting data cleaning")
        df_cleaned = df.copy()
        
        numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
        categorical_columns = [
            'gender', 'senior_citizen', 'is_married', 'dependents',
            'internet_service', 'contract', 'payment_method',
            'phone_service', 'dual', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv',
            'streaming_movies', 'paperless_billing'
        ]
        
        if is_training:
            # Calculate and store imputation values during training
            self.numerical_imputation_values = {
                col: df_cleaned[col].median() for col in numerical_columns if col in df_cleaned.columns
            }
            self.categorical_imputation_values = {
                col: df_cleaned[col].mode()[0] for col in categorical_columns if col in df_cleaned.columns
            }
            logger.info(f"Stored numerical imputation values: {self.numerical_imputation_values}")
            logger.info(f"Stored categorical imputation values: {self.categorical_imputation_values}")
        
        # Impute missing values only if necessary
        for col in numerical_columns:
            if col in df_cleaned.columns and df_cleaned[col].isna().any():
                impute_value = self.numerical_imputation_values.get(col)
                logger.info(f"Imputing missing values in {col} with {impute_value}")
                df_cleaned[col] = df_cleaned[col].fillna(impute_value)
            else:
                logger.debug(f"No missing values in {col}, skipping imputation")
                
        for col in categorical_columns:
            if col in df_cleaned.columns and df_cleaned[col].isna().any():
                impute_value = self.categorical_imputation_values.get(col)
                logger.info(f"Imputing missing values in {col} with {impute_value}")
                df_cleaned[col] = df_cleaned[col].fillna(impute_value)
            else:
                logger.debug(f"No missing values in {col}, skipping imputation")
                
        logger.info("Data cleaning completed")
        return df_cleaned


    def preprocess_data(self, df, is_training=True):
        """Preprocess the dataset for training or inference"""
        logger.info("Starting data preprocessing")
        # Validate and clean data
        df = self.validate_data(df, is_training=is_training)
        df_processed = self.clean_data(df, is_training=is_training)
        
        # Handle categorical variables
        self.categorical_columns = [
            'internet_service', 'contract', 'payment_method',
            'dual', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv',
            'streaming_movies', 
        ]
        
        binary_columns = ['gender','phone_service','paperless_billing','senior_citizen', 'is_married', 'dependents']
        # Map binary columns to 0/1
        logger.info("Mapping binary columns to 0/1")
        binary_mappings = {
            'gender': {'Male': 0, 'Female': 1, 'male': 0, 'female': 1},
            'phone_service': {'No': 0, 'Yes': 1, 'no': 0, 'yes': 1},
            'paperless_billing': {'No': 0, 'Yes': 1, 'no': 0, 'yes': 1},
            'senior_citizen': {'No': 0, 'Yes': 1, 'no': 0, 'yes': 1, 0: 0, 1: 1},
            'is_married': {'No': 0, 'Yes': 1, 'no': 0, 'yes': 1},
            'dependents': {'No': 0, 'Yes': 1, 'no': 0, 'yes': 1}
        }

        for col in binary_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map(binary_mappings[col])
                if df_processed[col].isna().any():
                    logger.warning(f"Missing or unmapped values in {col}, filling with 0")
                    df_processed[col] = df_processed[col].fillna(0)
        
        # Apply OneHotEncoder to categorical columns
        logger.info("Applying OneHotEncoder to categorical columns")
        if is_training:
            encoded_data = self.onehot_encoder.fit_transform(df_processed[self.categorical_columns])
            encoded_columns = self.onehot_encoder.get_feature_names_out(self.categorical_columns).tolist()
            logger.info(f"Feature columns after encoding: {encoded_columns}")
        else:
            encoded_data = self.onehot_encoder.transform(df_processed[self.categorical_columns])
            encoded_columns = self.onehot_encoder.get_feature_names_out(self.categorical_columns).tolist()
        
        # Create DataFrame with encoded categorical features
        df_encoded = pd.DataFrame(
            encoded_data,
            columns=encoded_columns,
            index=df_processed.index
        )
        
        # Handle numerical columns
        numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
        logger.info("Converting Total_Charges to numeric")
        df_processed['total_charges'] = pd.to_numeric(df_processed['total_charges'], errors='coerce')
        df_processed['total_charges']=df_processed['total_charges'].fillna(0)
        
        # FIXED: Preserve churn column during training
        columns_to_keep = numerical_columns + binary_columns
        if is_training and 'churn' in df_processed.columns:
            columns_to_keep.append('churn')
            logger.info("Preserving churn column for training")
        
        # Combine encoded categorical and numerical features
        logger.info("Combining numerical and encoded categorical features")
        df_processed = pd.concat([
            df_processed[columns_to_keep],
            df_encoded
        ], axis=1)
        
        if is_training:
            # Feature columns should not include the target variable
            self.feature_columns = numerical_columns + binary_columns + encoded_columns
            logger.info(f"Feature columns: {self.feature_columns}")
        
        logger.info("Data preprocessing completed")
        return df_processed
    

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained first")
        
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)


    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning using GridSearchCV with recall scoring"""
        logger.info("Starting hyperparameter tuning")
        param_grid = {
            'n_estimators': [ 30,50,70,100],
            'max_depth': [ 5,10,15,20,50],
            'min_samples_split': [ 20,30,50,70],
            'min_samples_leaf': [5,10,15,20,25],
            'class_weight': ['balanced']
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best recall score: {grid_search.best_score_:.3f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }


    def train(self, df):
        """Train the churn prediction model with cross-validation and reporting"""
        logger.info(f"Starting training with model: {self.model_name}")
        
        try:
            # Preprocess data
            logger.info("Preprocessing training data")
            df_processed = self.preprocess_data(df, is_training=True)
            
            # Verify churn column exists
            if 'churn' not in df_processed.columns:
                logger.error("Churn column not found after preprocessing")
                raise ValueError("Churn column not found after preprocessing")
            
            # Prepare features and target
            logger.info("Preparing features and target")
            X = df_processed[self.feature_columns]
            y = df_processed['churn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Target vector shape: {y.shape}")
            logger.info(f"Target distribution: {y.value_counts()}")
            
            # Split data
            logger.info("Splitting data into training and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Hyperparameter tuning
            logger.info("Performing hyperparameter tuning")
            tuning_results = self.hyperparameter_tuning(X_train, y_train)
            
            # Perform cross-validation
            logger.info("Performing cross-validation")
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=3, scoring='recall'
            )
            
            # Train final model
            logger.info("Training final model")
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            logger.info("Evaluating model on test set")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Extract precision, recall, and f1-score for class 1 (churn)
            precision = class_report['1']['precision']
            recall = class_report['1']['recall']
            f1 = class_report['1']['f1-score']
            
            # Generate training report
            self.training_report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': self.model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist()
                },
                'best_parameters': tuning_results['best_params'],
                'feature_importance': self.get_feature_importance(),
                'classification_report': class_report
            }
            
            logger.info(f"Training Report: {self.training_report}")
            logger.info("Training completed successfully")
            return self.training_report
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


    def predict_batch(self, customer_data, top_n=4):
        """Predict churn for multiple customers with probabilities and SHAP key factors"""
        logger.info(f"Starting batch prediction with model: {self.model_name}")

        try:
            if isinstance(customer_data, dict):
                df_input = pd.DataFrame([customer_data])
                logger.info("Received single dictionary input")
            elif isinstance(customer_data, list) and all(isinstance(item, dict) for item in customer_data):
                df_input = pd.DataFrame(customer_data)
                logger.info(f"Received list of {len(customer_data)} dictionaries")
            else:
                logger.error("Invalid input format")
                raise ValueError("Input must be a dictionary or a list of dictionaries")

            df_processed = self.preprocess_data(df_input, is_training=False)

            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = self.numerical_imputation_values.get(col, 0)

            X = df_processed[self.feature_columns]

            churn_preds = self.model.predict(X)
            churn_proba = self.model.predict_proba(X)[:, 1]

       
            results = []
            for i in range(len(X)):
                churn_label = "Yes" if churn_preds[i] == 1 else "No"
                prob = float(churn_proba[i])

                results.append({
                    "churn": churn_label,
                    "probability": prob
                })

            logger.info(f"Prediction results: {results}")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise


    def predict(self, customer_data):
        """Predict churn for a single customer"""
        logger.info(f"Starting single prediction with model: {self.model_name}")
        
        try:
            results = self.predict_batch(customer_data)
            result = results[0]
            
            logger.info("Single prediction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise
    

    def get_training_report(self):
        """Return the latest training report"""
        return self.training_report
    

    def save_model(self, filepath):
        """Save the trained model using pickle"""
        logger.info(f"Saving model {self.model_name} to {filepath}")
        
        try:
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'onehot_encoder': self.onehot_encoder,
                'feature_columns': self.feature_columns,
                'categorical_columns': self.categorical_columns,
                'training_report': self.training_report,
                'numerical_imputation_values': self.numerical_imputation_values,
                'categorical_imputation_values': self.categorical_imputation_values
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    
    def load_model(self, filepath):
        """Load a trained model using pickle"""
        logger.info(f"Loading model from {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.model_name = model_data.get('model_name', 'RandomForest')
            self.onehot_encoder = model_data['onehot_encoder']
            self.feature_columns = model_data['feature_columns']
            self.categorical_columns = model_data['categorical_columns']
            self.training_report = model_data.get('training_report', {})
            self.numerical_imputation_values = model_data.get('numerical_imputation_values', {})
            self.categorical_imputation_values = model_data.get('categorical_imputation_values', {})
            logger.info(f"Loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
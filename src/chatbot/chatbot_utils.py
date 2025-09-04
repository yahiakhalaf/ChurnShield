import glob
import os
from pydantic import BaseModel, Field
from src.logger_config import load_logger
logger=load_logger('chatbot_utils')

def get_latest_model_path(models_dir="models") -> str:
    """Find the most recent churn model file in the models directory."""
    logger.info(f"Searching for models in directory: {models_dir}")
    pattern = os.path.join(models_dir, "churn_model_*.pkl")
    model_files = glob.glob(pattern)

    if not model_files:
        logger.error(f"No models found in {models_dir} with pattern churn_model_*.pkl")
        raise FileNotFoundError(f"No models found in {models_dir}")

    latest_model = max(model_files, key=os.path.getctime)
    logger.info(f"Found {len(model_files)} model(s), selected latest: {latest_model}")
    return latest_model


class CustomerData(BaseModel):
    gender: str = Field(None, description="Customer gender (Male/Female)")
    senior_citizen: int = Field(None, description="Whether the customer is a senior citizen (0 or 1)")
    is_married: str = Field(None, description="Whether the customer is married (Yes/No)")
    dependents: str = Field(None, description="Whether the customer has dependents (Yes/No)")
    tenure: float = Field(None, description="Number of months since the customer joined")
    internet_service: str = Field(None, description="Internet service provider (DSL/Fiber optic/No)")
    contract: str = Field(None, description="Contract duration (Month-to-month/One year/Two years)")
    payment_method: str = Field(None, description="Payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))")
    monthly_charges: float = Field(None, description="Monthly charges amount")
    total_charges: float = Field(None, description="Total charges amount")
    phone_service: str = Field(None, description="Whether the customer has phone service (Yes/No)")
    dual: str = Field(None, description="Whether the customer is a dual customer (Yes/No/No phone service)")
    online_security: str = Field(None, description="Whether the customer has online security (Yes/No/No internet service)")
    online_backup: str = Field(None, description="Whether the customer has online backup (Yes/No/No internet service)")
    device_protection: str = Field(None, description="Whether the customer has device protection (Yes/No/No internet service)")
    tech_support: str = Field(None, description="Whether the customer receives tech support (Yes/No/No internet service)")
    streaming_tv: str = Field(None, description="Whether the customer has streaming TV (Yes/No/No internet service)")
    streaming_movies: str = Field(None, description="Whether the customer has streaming movies (Yes/No/No internet service)")
    paperless_billing: str = Field(None, description="Whether the customer has paperless billing (Yes/No)")




class PredictionResult(BaseModel):
    churn: str = Field(..., description="Predicted churn status (Yes/No)")
    probability: float = Field(..., description="Probability of churn")

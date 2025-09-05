from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, Optional
from src.chatbot.chatbot import ChurnChatbot
from src.chatbot.chatbot_utils import get_latest_model_path
from src.logger_config import load_logger 

logger = load_logger("FastAPI")

app = FastAPI(
    title="Churn Prediction Chatbot API",
    description="API for interacting with the Churn Prediction Chatbot",
    version="1.0.0"
)

# Initialize chatbot
try:
    model_path = get_latest_model_path("models")
    chatbot = ChurnChatbot(model_path)
    logger.info("FastAPI application initialized with ChurnChatbot")
except Exception as e:
    logger.error(f"Failed to initialize ChurnChatbot: {str(e)}")
    raise Exception(f"Failed to initialize ChurnChatbot: {str(e)}")

class MessageRequest(BaseModel):
    message: str

class SessionInfoResponse(BaseModel):
    has_customer_data: bool
    has_prediction: bool
    customer_fields: int
    last_prediction_summary: Optional[Dict[str, Any]] 

class MessageResponse(BaseModel):
    response: str

@app.post("/message", response_model=MessageResponse)
async def handle_message(request: MessageRequest):
    """
    Process a user message and return the chatbot's response.
    
    Args:
        request: MessageRequest containing the user's message
        
    Returns:
        MessageResponse containing the chatbot's response
    """
    try:
        logger.info(f"Processing message: {request.message[:100]}...")
        response = chatbot.handle_message(request.message)
        return MessageResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        error_msg = "I encountered an issue processing your message. Please try again or rephrase your question."
        return MessageResponse(response=error_msg)

@app.get("/session_info", response_model=SessionInfoResponse)
async def get_session_info():
    """
    Get information about the current chatbot session.
    
    Returns:
        SessionInfoResponse containing session details
    """
    try:
        logger.info("Retrieving session information")
        session_info = chatbot.get_session_info()
        return SessionInfoResponse(**session_info)
    except Exception as e:
        logger.error(f"Error retrieving session info: {str(e)}")
        # Return default session info on error
        return SessionInfoResponse(
            has_customer_data=False,
            has_prediction=False,
            customer_fields=0,
            last_prediction_summary=None
        )

@app.post("/clear_memory")
async def clear_memory():
    """
    Clear the chatbot's memory and reset interaction counter.
    
    Returns:
        MessageResponse confirming memory clearance
    """
    try:
        logger.info("Clearing chatbot memory")
        chatbot.clear_memory()
        return MessageResponse(response="Memory cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        return MessageResponse(response="Error clearing memory. Please try again.")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        Dict confirming API status
    """
    logger.info("Health check requested")
    return {"status": "healthy", "service": "Churn Prediction Chatbot API"}

@app.get("/")
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "Churn Prediction Chatbot API",
        "version": "1.0.0",
        "endpoints": ["/message", "/session_info", "/clear_memory", "/health"]
    }

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
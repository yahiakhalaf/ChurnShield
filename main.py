import multiprocessing
import uvicorn
from src.logger_config import setup_logger
from src.gradio import create_interface  

logger = setup_logger("Main")

def run_fastapi():
    """Run the FastAPI server from fast_api.py."""
    try:
        logger.info("Starting FastAPI server")
        uvicorn.run(
            "src.fast_api:app",  
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"FastAPI server failed: {str(e)}")
        raise

def run_gradio():
    """Run the Gradio interface from gradio.py."""
    try:
        logger.info("Starting Gradio interface")
        demo = create_interface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,  
            show_error=True
        )
    except Exception as e:
        logger.error(f"Gradio interface failed: {str(e)}")
        raise

def start_application():
    """Start the full application (FastAPI and Gradio) in separate processes."""
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    gradio_process = multiprocessing.Process(target=run_gradio)

    try:
        logger.info("Starting Churn Prediction Chatbot application (FastAPI + Gradio)")
        fastapi_process.start()
        gradio_process.start()

        # Wait for both processes to complete
        fastapi_process.join()
        gradio_process.join()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, terminating processes")
        fastapi_process.terminate()
        gradio_process.terminate()
        fastapi_process.join()
        gradio_process.join()
        logger.info("Application terminated successfully")
    except Exception as e:
        logger.error(f"Error in application startup: {str(e)}")
        fastapi_process.terminate()
        gradio_process.terminate()
        raise

if __name__ == "__main__":
    logger.info("Initializing Churn Prediction Chatbot application")
    start_application()

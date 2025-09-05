import gradio as gr
import requests
from typing import List, Dict
from src.logger_config import load_logger 
import yaml

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

logger = load_logger("gradio UI")

# Configuration
API_BASE_URL = config["fastapi"]["api_base_url"]


def test_connection() -> bool:
    """Test if the FastAPI backend is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("FastAPI backend is reachable.")
            return True
        else:
            logger.warning(f"FastAPI health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error: {str(e)}")
        return False


def send_message(message: str) -> str:
    """Send message to FastAPI backend and return response."""
    try:
        logger.info(f"Sending message to backend: {message[:50]}...")
        response = requests.post(
            f"{API_BASE_URL}/message",
            json={"message": message},
            timeout=30
        )
        
        if response.status_code == 200:
            reply = response.json()["response"]
            logger.info(f"Received response: {reply[:50]}...")
            return reply
        else:
            error_detail = response.json().get("detail", "Unknown error")
            logger.error(f"Backend returned error: {error_detail}")
            return f"Error: {error_detail}"
            
    except requests.exceptions.Timeout:
        logger.error("Request timed out.")
        return "Error: Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to backend.")
        return "Error: Cannot connect to the chatbot service."
    except Exception as e:
        logger.exception(f"Unexpected error while sending message: {str(e)}")
        return f"Error: {str(e)}"


def chat_interface(message: str, history: List[Dict]) -> tuple[List[Dict], str]:
    """Main chat interface function using OpenAI-style messages."""
    if not message.strip():
        logger.warning("Empty message received from user.")
        return history, ""
    
    if not test_connection():
        error_msg = "Cannot connect to the chatbot service."
        logger.error(error_msg)
        history.append({"role": "assistant", "content": error_msg})
        return history, ""
    
    response = send_message(message)
    
    # Append user + assistant messages
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    logger.info("Chat updated with new interaction.")
    return history, ""


def load_sample_data():
    """Load sample customer data for testing."""
    logger.debug("Loading sample test data.")
    return {
        "High Risk Customer": "Customer John is a 35-year-old male, married with dependents. He has been with us for 6 months, pays $85 monthly with total charges of $510. He has fiber optic internet, phone service, and streaming services but no online security or tech support. He uses electronic check payment and paperless billing with a month-to-month contract.",
        "Low Risk Customer": "Customer Sarah is a 45-year-old female, married with dependents. She has been loyal for 48 months, pays $65 monthly with total charges of $3120. She has DSL internet with all security features, phone service, and uses automatic bank transfer. She has a two-year contract.",
        "Medium Risk Customer": "Customer Mike is a 28-year-old male, single with no dependents. He's been with us for 18 months, pays $75 monthly with total charges of $1350. He has fiber optic internet with streaming services but no security features. He pays by credit card with a one-year contract."
    }


def update_connection_status():
    """Update connection status display."""
    status = "Connected to FastAPI backend" if test_connection() else "Cannot connect to FastAPI backend"
    logger.info(f"Backend status: {status}")
    return status


def set_sample_text(sample_key: str) -> str:
    """Return the sample text for a given key."""
    samples = load_sample_data()
    text = samples.get(sample_key, "")
    logger.info(f"Loaded sample: {sample_key}")
    return text


# Create the Gradio interface
def create_interface():
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .chat-message {
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Churn Prediction Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎯 Customer Churn Prediction Chatbot
            
            This AI-powered chatbot helps predict customer churn and provides retention recommendations.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat with the Churn Prediction Bot",
                    height=500,
                    show_copy_button=True,
                    elem_classes=["chat-message"],
                    type="messages" 
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me anything about churn prediction or provide customer details...",
                        label="Your Message",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Sample data section
                gr.Markdown("### 📝 Quick Test Examples")
                with gr.Row():
                    high_risk_btn = gr.Button("High Risk Customer", size="sm")
                    low_risk_btn = gr.Button("Low Risk Customer", size="sm")
                    medium_risk_btn = gr.Button("Medium Risk Customer", size="sm")
            
            with gr.Column(scale=1):
                # Connection status
                gr.Markdown("### 🔌 Connection Info")
                connection_status = gr.Textbox(
                    label="Backend Status",
                    value="Checking connection...",
                    interactive=False,
                    lines=2
                )
                
                # Usage tips
                gr.Markdown(
                    """
                    ### 💡 Usage Tips
                    
                    **For Predictions:**
                    - Include customer details like tenure, charges, services
                    - Be specific about contract type and payment method
                    
                    **For Follow-up:**
                    - Ask "What can we do to retain this customer?"
                    - Request "detailed recommendations"
                    
                    **For Corrections:**
                    - Say "Actually, the tenure should be X months"
                    - Specify what needs to be changed
                    """
                )
        
        # Event handlers
        send_btn.click(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        # Sample buttons
        high_risk_btn.click(lambda: set_sample_text("High Risk Customer"), outputs=msg)
        low_risk_btn.click(lambda: set_sample_text("Low Risk Customer"), outputs=msg)
        medium_risk_btn.click(lambda: set_sample_text("Medium Risk Customer"), outputs=msg)
        
        # Initialize connection status on load
        demo.load(update_connection_status, outputs=connection_status)
    
    return demo

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo = create_interface()
    demo.launch(
        server_name=config["gradio"]["server_name"],
        server_port=config["gradio"]["server_port"],
        share=config["gradio"]["share"],
        debug=config["gradio"]["debug"],
        show_error=config["gradio"]["show_error"]
    )

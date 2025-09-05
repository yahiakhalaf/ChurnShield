# ChurnShield: AI-Powered Customer Churn Prediction Chatbot

ChurnShield is an intelligent customer churn prediction system that combines machine learning with conversational AI to help businesses identify at-risk customers and develop retention strategies.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Chatbot Capabilities](#chatbot-capabilities)
- [Customer Data Fields](#customer-data-fields)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

## Description

ChurnShield addresses one of the most critical challenges in business: customer retention. By leveraging advanced machine learning algorithms and natural language processing, the system transforms complex customer data analysis into intuitive conversations.

The application operates on a sophisticated multi-layered architecture. At its core lies a Random Forest classifier trained with comprehensive hyperparameter optimization to maximize recall for churn detection. This model processes 19 customer attributes ranging from demographic information to service usage patterns, generating probability scores that indicate churn likelihood.

What sets ChurnShield apart is its conversational interface powered by Ollama's LLaMA 3.1 model. Instead of requiring users to input structured data through forms, the system understands natural language descriptions of customers. Users can describe a customer scenario in plain English, and the AI extracts relevant features, processes corrections, and provides contextual recommendations.

The system maintains conversation memory, allowing for dynamic data correction and follow-up analysis. When a user says "Actually, the tenure should be 24 months," the system intelligently updates the customer profile and regenerates predictions without losing context.

ChurnShield provides three interaction modalities: a user-friendly Gradio web interface for business users, a comprehensive FastAPI REST API for system integration, and a command-line interface for developers and analysts. Each interface maintains consistent functionality while optimizing for different use cases.

The application implements robust logging and monitoring systems, tracking model performance, user interactions, and system health. This enables continuous improvement and reliable operation in production environments.

## Features

- **AI-Powered Churn Prediction**: Random Forest model with hyperparameter tuning for accurate churn predictions
- **Conversational Interface**: Natural language chatbot powered by Ollama LLM for intuitive customer data input
- **Dual Interface**: Both FastAPI REST API and Gradio web interface
- **Real-time Analysis**: Instant churn probability assessment with actionable recommendations
- **Memory Management**: Context-aware conversations with automatic memory cleanup
- **Comprehensive Logging**: Detailed logging system for monitoring and debugging

## Architecture

```
ChurnShield/
├── main.py                 # Application entry point
├── config.yaml            # Configuration settings
├── pyproject.toml         # Python dependencies
└── src/
    ├── fast_api.py                 # FastAPI REST API server
    ├── gradio.py                   # Gradio web interface
    ├── logger_config.py            # Logging configuration
    ├── churn_model/
    │   ├── churn_predictor.py      # ML model implementation
    │   └── model_training.py       # Model training pipeline
    └── chatbot/
        ├── chatbot.py              # Main chatbot logic
        └── chatbot_utils.py        # Utility classes and functions
```

## Prerequisites

- Python 3.12+
- Customer churn dataset (CSV format)
- At least 4GB RAM (for Ollama model)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yahiakhalaf/ChurnShield.git
cd ChurnShield
```

### 2. Install UV Package Manager

UV is a fast Python package installer and resolver. Install it using one of these methods:

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

### 3. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install project dependencies
uv pip install -e .
```

### 4. Install Ollama

Ollama is required to run the LLM for natural language processing.

**On macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.ai/download
```

**On Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**On Windows:**
```bash
# Download installer from https://ollama.ai/download
# Run the installer and follow instructions
```

### 5. Setup Ollama Model

```bash
# Start Ollama service (in background)
ollama serve

# In a new terminal, pull the required model
ollama pull llama3.1:8b
```

Note: The `llama3.1:8b` model is approximately 4.7GB and may take some time to download.

### 6. Prepare Dataset

Place your customer churn dataset (CSV format) in the `datasets/` directory:
```bash
mkdir -p datasets
# Copy your Customer-Churn.csv file to datasets/
```

## Usage

### Train the Model

Before using the chatbot, train the churn prediction model:

```bash
uv run python -m src.churn_model.model_training --input datasets/Customer-Churn.csv
```

This will:
- Train a Random Forest model with hyperparameter optimization
- Save the trained model to `models/churn_model_<timestamp>.pkl`
- Generate a training report in `reports/training_report_<timestamp>.json`

### Start the Application

Launch both FastAPI and Gradio interfaces:

```bash
uv run python main.py
```

This starts:
- **FastAPI API**: http://localhost:8000 (Interactive docs at http://localhost:8000/docs)
- **Gradio Interface**: http://localhost:7860

### Interface Options

#### 1. Web Interface (Recommended)

Open http://localhost:7860 in your browser:

1. Use sample customer data buttons for quick testing
2. Or describe a customer naturally: "Customer John is 35 years old, has been with us for 6 months..."
3. Get instant churn predictions with probability scores
4. Ask follow-up questions for retention recommendations

#### 2. API Interface

**Predict Churn:**
```bash
curl -X POST "http://localhost:8000/message" \
     -H "Content-Type: application/json" \
     -d '{"message": "Customer Sarah is 45, married with dependents, 48 months tenure, pays $65 monthly"}'
```

**Get Session Info:**
```bash
curl -X GET "http://localhost:8000/session_info"
```

**Clear Memory:**
```bash
curl -X POST "http://localhost:8000/clear_memory"
```

#### 3. Command Line Interface

Run the chatbot directly:
```bash
uv run python -m src.chatbot.chatbot
```

## Configuration

Customize settings in `config.yaml`:

```yaml
# Server settings
fastapi:
  host: "0.0.0.0"
  port: 8000

gradio:
  server_port: 7860

# Model settings
churn_predictor:
  model_name: "RandomForest"
  cv_folds: 3
  scoring_metric: "recall"
  hyperparameter_grid:
    n_estimators: [30, 50, 70, 100]
    max_depth: [5, 10, 15, 20, 50]

# Chatbot settings
chatbot:
  memory_threshold: 10
  model_name: "llama3.1:8b"
```

## Chatbot Capabilities

### Intent Classification
The chatbot automatically classifies user messages:
- **General**: Greetings, general questions about churn
- **Client**: New customer data for prediction
- **Correction**: Updates to previously provided data
- **Followup**: Questions about predictions and recommendations

### Sample Interactions

**Providing Customer Data:**
```
User: "Customer Mike is 28, single, 18 months with us, pays $75/month, has fiber internet but no security features"
Bot: "Churn Prediction Results
     • Prediction: Yes
     • Probability: 73%"
```

**Getting Recommendations:**
```
User: "What can we do to retain this customer?"
Bot: "Based on the high churn risk (73%), I recommend:
     1. Immediate: Offer online security package at discount
     2. Long-term: Upgrade to annual contract with incentives..."
```

**Correcting Data:**
```
User: "Actually, the tenure should be 24 months, not 18"
Bot: "Customer Data Updated Successfully
     New Prediction Results:
     • Prediction: No  
     • Probability: 42%"
```

## Customer Data Fields

The system recognizes these customer attributes:

| Field | Type | Values | Description |
|-------|------|---------|-------------|
| gender | String | Male/Female | Customer gender |
| senior_citizen | Integer | 0/1 | Senior citizen status |
| is_married | String | Yes/No | Marital status |
| dependents | String | Yes/No | Has dependents |
| tenure | Float | Number | Months with company |
| internet_service | String | DSL/Fiber optic/No | Internet service type |
| contract | String | Month-to-month/One year/Two years | Contract duration |
| payment_method | String | Electronic check/Mailed check/Bank transfer/Credit card | Payment method |
| monthly_charges | Float | Number | Monthly charge amount |
| total_charges | Float | Number | Total charges to date |
| phone_service | String | Yes/No | Phone service subscription |
| dual | String | Yes/No/No phone service | Multiple lines |
| online_security | String | Yes/No/No internet service | Security service |
| online_backup | String | Yes/No/No internet service | Backup service |
| device_protection | String | Yes/No/No internet service | Device protection |
| tech_support | String | Yes/No/No internet service | Technical support |
| streaming_tv | String | Yes/No/No internet service | TV streaming |
| streaming_movies | String | Yes/No/No internet service | Movie streaming |
| paperless_billing | String | Yes/No | Paperless billing preference |

## Development

### Running Tests
```bash
# Run specific module
uv run python -m src.churn_model.churn_predictor

# Train model with custom parameters
uv run python -m src.churn_model.model_training --input datasets/custom.csv --model-dir models/
```

### Adding Dependencies
```bash
# Add new dependencies
uv add package-name

# Add development dependencies
uv add --dev pytest black flake8
```

### Logging

Logs are automatically saved to `logs/ChurnShield_<timestamp>.log` with detailed information about:
- Model training progress
- Prediction requests and results
- Error handling and debugging
- API request processing

## Troubleshooting

### Common Issues

**Ollama Connection Error:**
- Ensure Ollama is running: `ollama serve`
- Check if model is installed: `ollama list`
- Verify model name in config matches installed model

**Model Not Found:**
- Train the model first: `uv run python -m src.churn_model.model_training`
- Check `models/` directory for `.pkl` files

**Port Already in Use:**
- Change ports in `config.yaml`
- Kill existing processes: `lsof -ti:8000 | xargs kill -9`

**Memory Issues:**
- Ensure at least 4GB RAM available
- Close other applications if needed
- Consider using smaller Ollama models

### Performance Tips

1. **Model Training**: Use smaller hyperparameter grids for faster training
2. **Memory Management**: Adjust `memory_threshold` in config for frequent conversations
3. **API Performance**: Use FastAPI for high-throughput scenarios
4. **Model Size**: Consider `llama3.1:3b` for lower resource usage

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ollama for providing local LLM capabilities
- FastAPI for the robust API framework
- Gradio for the intuitive web interface
- scikit-learn for machine learning tools

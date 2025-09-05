import json
import time
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.churn_model.churn_predictor import ChurnPredictor
from src.logger_config import load_logger
from src.chatbot.chatbot_utils import CustomerData,PredictionResult,get_latest_model_path
from src.churn_model.model_training import train_and_save_model
import yaml

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

logger = load_logger("ChurnChatbot")


class ChurnChatbot:
    def __init__(self, model_path: str):
        logger.info("Initializing ChurnChatbot...")
        start_time = time.time()
        
        try:
            # Initialize churn predictor
            logger.info("Loading churn predictor model...")
            self.predictor = ChurnPredictor()
            self.predictor.load_model(model_path)            
            logger.info("Churn predictor model loaded successfully")
            self.llm = ChatOllama(model=config["chatbot"]["model_name"])

            # Initialize LLM for general text responses (no JSON format)
            logger.info("Initializing Ollama LLM for general responses...")
            self.llm_general =  self.llm
            
            # Initialize LLM for structured JSON responses
            logger.info("Initializing Ollama LLM for JSON responses...")
            self.llm_json = self.llm.with_structured_output(CustomerData)
            
            logger.info("Ollama LLMs initialized successfully")

            # Initialize memory and parser
            self.memory = {"last_customer": None, "last_prediction": None}
            self.interaction_count = 0  
            self.memory_threshold =config["chatbot"]["memory_threshold"]
            init_time = time.time() - start_time
            logger.info(f"ChurnChatbot initialization completed in {init_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChurnChatbot: {str(e)}", exc_info=True)
            raise



    def process_general_query(self, message: str) -> str:
        """Handle general knowledge questions and non-churn related queries."""
        logger.info(f"Processing general query: {message[:100]}...")
        start_time = time.time()
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly and knowledgeable AI assistant with expertise in customer churn prevention and general knowledge.

For general questions like greetings, geography, science, history, etc., provide natural, conversational responses.
For churn-related questions, provide professional insights about customer retention and churn analysis.

Keep your responses conversational, helpful, and engaging. Respond naturally as if you're having a friendly conversation.

Examples:
- "How are you?" → "I'm doing great, thanks for asking! How can I help you today?"
- "What is the capital of Egypt?" → "The capital of Egypt is Cairo, a fascinating city with rich history!"
- "What is churn?" → "Customer churn refers to when customers stop using a company's services..."

Always be friendly and helpful."""),
                ("human", "{msg}")
            ])
            
            chain = prompt | self.llm_general
            response = chain.invoke({"msg": message})
            
            processing_time = time.time() - start_time
            logger.info(f"General query processed in {processing_time:.2f} seconds")
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing general query: {str(e)}", exc_info=True)
            return f"I'm doing well, thanks for asking! How can I help you with customer churn analysis or any other questions you might have?"


    def clear_memory(self):
            """Clear the memory dictionary and reset interaction counter."""
            logger.info("Clearing memory due to threshold or manual reset")
            self.memory = {"last_customer": None, "last_prediction": None}
            self.interaction_count = 0

    def process_client_query(self, message: str) -> str:
        """Extract customer data and generate churn prediction."""
        logger.info(f"Processing client query: {message[:100]}...")
        start_time = time.time()
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a data extraction expert with advanced contextual understanding. Extract customer information from natural language text, interpreting indirect references and contextual clues for complex extractions.

            Extract ONLY the following customer attributes and return them as valid JSON:
            {list(CustomerData.model_fields.keys())}

            Guidelines:
            - Use exact values: "Yes"/"No" for boolean fields (is_married, dependents, phone_service, paperless_billing), exact numbers for numeric fields (tenure, monthly_charges, total_charges)
            - For gender: use "Male" or "Female" based on explicit mention or context (e.g., "he" implies "Male", "she" implies "Female")
            - For senior_citizen: use 0 (No) or 1 (Yes) based on age context (e.g., "older client" or "over 65" implies 1, "young" implies 0)
            - For online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies: "Yes", "No", or "No internet service".
            - For dual: "Yes", "No", or "No phone service".
            - For contract: use "Month-to-month", "One year", or "Two years" based on terms like "short-term" (Month-to-month), "annual" (One year), or "long-term" (Two years)
            - For internet_service: use "DSL", "Fiber optic", or "No" based on mentions like "high-speed internet" (Fiber optic), "basic connection" (DSL), or "no internet" (No)
            - For payment_method: use "Electronic check", "Postal check", "Bank transfer", or "Credit card" based on phrases like "online payment" (Electronic check), "mail payment" (Postal check), "auto-pay" (Bank transfer/Credit card with context)
            - If information is not provided  make it None (absence of value)
            - Return valid JSON only, no additional text, ensuring proper formatting and no trailing commas"""),
                ("human", "Extract customer data from: {msg}")
            ])

            chain = prompt | self.llm_json
            customer: CustomerData = chain.invoke({"msg": message})
            
            customer_dict = customer.model_dump()
            for key, value in customer_dict.items():
                if value == 'None':
                    customer_dict[key] = None
            self.memory["last_customer"] = customer_dict
            logger.info(f"Extracted customer data: {customer_dict}")

            self.interaction_count += 1
            if self.interaction_count >= self.memory_threshold:
                self.clear_memory()
            # Generate prediction
            prediction_start = time.time()
            prediction_dict = self.predictor.predict(customer_dict)
            prediction = PredictionResult(**prediction_dict)
            self.memory["last_prediction"] = prediction
            
            prediction_time = time.time() - prediction_start
            logger.info(f"Churn prediction generated in {prediction_time:.2f} seconds: {prediction.churn} ({prediction.probability:.2%})")

            
            total_time = time.time() - start_time
            logger.info(f"Client query processed in {total_time:.2f} seconds")
            
            return (f" **Churn Prediction Results**\n"
                   f"• **Prediction**: {prediction.churn}\n"
                   f"• **Probability**: {prediction.probability:.2%}\n"
                   f" Ask me for detailed recommendations or provide corrections if needed.")

        except Exception as e:
            logger.error(f"Error processing client query: {str(e)}", exc_info=True)
            return f" Could not process customer data. Please provide clear customer information with details like tenure, charges, services, etc."



    def process_followup_query(self, message: str) -> str:
        """Provide detailed analysis and recommendations based on the last prediction."""
        logger.info(f"Processing followup query: {message[:100]}...")
        
        if not self.memory["last_prediction"]:
            logger.warning("Followup query attempted without prior prediction")
            return " Please provide customer information first to get a churn prediction, then I can answer follow-up questions."

        start_time = time.time()
        prediction = self.memory["last_prediction"]
        customer_data = self.memory["last_customer"]
        
        try:
            customer_summary = ", ".join(f"{k}: {v}" for k, v in customer_data.items())
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a customer retention specialist providing expert advice based on churn prediction analysis.

**Current Customer Analysis:**
• Churn Prediction: {prediction.churn}
• Churn Probability: {prediction.probability:.2%}


**Customer Profile:**
{customer_summary}

Provide actionable, specific recommendations for customer retention. Include:
1. Immediate actions to reduce churn risk
2. Long-term strategies
3. Personalized offers or interventions
4. Monitoring recommendations

Be professional, data-driven, and practical."""),
                ("human", "Question: {msg}")
            ])
            
            chain = prompt | self.llm_general
            response = chain.invoke({"msg": message})

            self.interaction_count += 1
            if self.interaction_count >= self.memory_threshold:
                self.clear_memory()

            processing_time = time.time() - start_time
            logger.info(f"Followup query processed in {processing_time:.2f} seconds")
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error processing followup query: {str(e)}", exc_info=True)
            return f" I encountered an error while analyzing the customer data. Please try asking your question differently."



    def apply_correction(self, message: str) -> str:
        """Apply corrections to the last customer data and regenerate prediction."""
        logger.info(f"Processing correction: {message[:100]}...")
        
        if not self.memory["last_customer"]:
            logger.warning("Correction attempted without existing customer data")
            return "ℹ No customer data to correct. Please provide customer details first."

        start_time = time.time()
        try:
            current_data = self.memory["last_customer"]
            logger.info(f"Current customer data before correction: {current_data}")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a data correction specialist. Update the existing customer data with the corrections provided.

**Current Customer Data:**
{json.dumps(current_data, indent=2)}

**Instructions:**
- Keep all existing fields unchanged unless specifically mentioned in the correction
- Apply only the corrections mentioned in the user's message
- Maintain the same data format and field names
- Return the complete updated JSON with all fields (changed and unchanged)
- Use proper data types and valid values as per the original schema"""),
                ("human", "Apply these corrections: {msg}")
            ])

            chain = prompt | self.llm_json.with_structured_output(CustomerData)
            corrected: CustomerData = chain.invoke({"msg": message})
            # Update memory
            corrected_dict = corrected.model_dump()
            self.memory["last_customer"] = corrected_dict
            logger.info(f"Customer data after correction: {corrected_dict}")
            
            # Regenerate prediction with corrected data
            prediction_dict = self.predictor.predict(corrected_dict)
            prediction = PredictionResult(**prediction_dict)
            self.memory["last_prediction"] = prediction

            self.interaction_count += 1
            if self.interaction_count >= self.memory_threshold:
                self.clear_memory()
                
            total_time = time.time() - start_time
            logger.info(f"Correction applied and prediction updated in {total_time:.2f} seconds")
            
            
            return (f" **Customer Data Updated Successfully**\n\n"
                   f" **New Prediction Results:**\n"
                   f"• **Prediction**: {prediction.churn}\n"
                   f"• **Probability**: {prediction.probability:.2%}\n"
                   f" Updated customer profile: {len(corrected_dict)} fields")
            
        except Exception as e:
            logger.error(f"Error applying correction: {str(e)}", exc_info=True)
            return f" Could not apply the correction. Please specify clearly which customer attributes need to be changed."



    def classify_intent(self, message: str) -> str:
        """Classify user intent with enhanced prompt."""
        logger.debug(f"Classifying intent for message: {message[:50]}...")
        
        try:
            classifier_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an intent classifier for a customer churn prediction chatbot.

Classify the user's message into ONE of these categories:

**general**: General knowledge questions (geography, science, history, greetings, casual conversation, etc.) OR general churn/business questions not about a specific customer
**client**: New customer information provided for churn prediction (contains specific customer details like tenure, charges, services, etc.)
**correction**: User wants to correct, change,update or modifiy previously provided customer information
**followup**: Questions about the last prediction results, asking for recommendations, explanations, or analysis
**quit**: User wants to exit/quit/stop the conversation

Examples:
- "How are you?" → general
- "Hello" → general
- "What is the capital of France?" → general
- "How does churn prediction work?" → general  
- "Customer John has 24 months tenure, $80 monthly charges..." → client
- "Actually, the tenure should be 36 months" → correction
- "What can we do to retain this customer?" → followup
- "bye"or "exist" or "i will leave now" → quit

Return ONLY the classification label, nothing else."""),
                ("human", "{msg}")
            ])

            chain = classifier_prompt | self.llm_general
            response = chain.invoke({"msg": message})
            
            # Extract just the intent from the response
            intent_text = response.content if hasattr(response, 'content') else str(response)
            intent = intent_text.strip().lower()
            
            logger.debug(f"Classified intent: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}", exc_info=True)
            return "general"  



    def handle_message(self, message: str) -> str:
        """Main message handler with enhanced logging and error handling."""
        logger.info(f"Received message: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        if not message.strip():
            return " Please provide a message or question."
        
        try:
            # Classify intent
            intent = self.classify_intent(message)
            logger.info(f"Message classified as: {intent}")
            
            # Route to appropriate handler
            if "quit" in intent:
                logger.info("User requested to quit")
                return " Thank you for using the Churn Prediction Chatbot. Goodbye!"
            elif "general" in intent:
                return self.process_general_query(message)
            elif "client" in intent:
                return self.process_client_query(message)
            elif "correction" in intent:
                return self.apply_correction(message)
            elif "followup" in intent:
                return self.process_followup_query(message)
            else:
                logger.warning(f"Unknown intent classification: {intent}")
                return (" I'm not sure how to help with that. You can:\n"
                       "• Ask general questions (e.g., 'What is churn prediction?')\n"
                       "• Provide customer data for prediction\n"
                       "• Ask for recommendations about the last prediction\n"
                       "• Correct previous customer information")
                       
        except Exception as e:
            logger.error(f"Unexpected error in handle_message: {str(e)}", exc_info=True)
            return " I encountered an unexpected error. Please try again or rephrase your message."



    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "has_customer_data": self.memory["last_customer"] is not None,
            "has_prediction": self.memory["last_prediction"] is not None,
            "customer_fields": len(self.memory["last_customer"]) if self.memory["last_customer"] else 0,
            "last_prediction_summary": {
                "churn": self.memory["last_prediction"].churn,
                "probability": f"{self.memory['last_prediction'].probability:.2%}"
            } if self.memory["last_prediction"] else None
        }



    def interactive_loop(self):
        """Interactive command-line interface."""
        print(" Welcome to the Enhanced Churn Prediction Chatbot!")
        print(" I can help with:")
        print("   • General questions (geography, churn analysis, etc.)")
        print("   • Customer churn predictions")
        print("   • Retention recommendations")
        print("   • Data corrections")
        print("\n Type 'quit' to exit.")
        
        logger.info("Starting interactive loop")
        
        while True:
            try:
                query = input("\n You: ").strip()
                if not query:
                    continue
                    
                response = self.handle_message(query)
                print(f"\n🤖 Bot: {response}")
                
                if "Thank you for using" in response:
                    logger.info("Interactive session ended by user")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Interactive session interrupted by user")
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {str(e)}", exc_info=True)
                print(f"\n⚠️ An error occurred: {str(e)}")



if __name__ == "__main__":
    try:
        logger.info("Starting ChurnChatbot application")
        latest_model_path = get_latest_model_path("models")
        
        chatbot = ChurnChatbot(latest_model_path)
        chatbot.interactive_loop()
        
    except Exception as e:
        logger.error(f"Fatal error starting application: {str(e)}", exc_info=True)
        print(f"❌ Failed to start chatbot: {str(e)}")
        exit(1)
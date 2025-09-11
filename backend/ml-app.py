import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

# Set tracking URI and experiment
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))

import os
import requests
import time
import mlflow
from dotenv import load_dotenv

load_dotenv()

# MLflow setup (using your existing config)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "my-genai-experiment"))

# llama.cpp server configuration
LLAMA_SERVER_URL = "http://localhost:8080"  # Default llama.cpp server port

def query_llama(prompt, model_params=None):
    """Query your local llama.cpp server and track with MLflow"""
    
    # Default parameters if none provided
    if model_params is None:
        model_params = {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9
        }
    
    with mlflow.start_run():
        # Log the input parameters
        mlflow.log_param("prompt", prompt)
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        # Track timing
        start_time = time.time()
        
        try:
            # Make request to llama.cpp server
            response = requests.post(
                f"{LLAMA_SERVER_URL}/completion",
                json={
                    "prompt": prompt,
                    **model_params
                }
            )
            response.raise_for_status()
            
            # Extract response data
            result = response.json()
            completion_text = result.get("content", "")
            
            # Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            token_count = len(completion_text.split())  # Simple word count
            
            # Log metrics to MLflow
            mlflow.log_metric("response_time_seconds", response_time)
            mlflow.log_metric("estimated_tokens", token_count)
            mlflow.log_metric("tokens_per_second", token_count / response_time if response_time > 0 else 0)
            
            # Log the actual response
            mlflow.log_text(completion_text, "completion.txt")
            
            print(f"✓ Logged run with {token_count} tokens in {response_time:.2f}s")
            return completion_text
            
        except requests.exceptions.RequestException as e:
            mlflow.log_param("error", str(e))
            print(f"❌ Error connecting to llama.cpp server: {e}")
            return None

# Example usage
if __name__ == "__main__":
    prompt = "Explain recursion in programming"
    
    # Test with different parameters
    params = {
        "temperature": 0.3,
        "max_tokens": 100,
        "top_p": 0.95
    }
    
    response = query_llama(prompt, params)
    if response:
        print("Response:", response[:100] + "..." if len(response) > 100 else response)
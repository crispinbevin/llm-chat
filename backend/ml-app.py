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


def query_llama(user_prompt, system_prompt=None, model_params=None):
    """Query your local llama.cpp server with optional system prompt and track with MLflow"""
    
    # Default parameters if none provided
    if model_params is None:
        model_params = {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9
        }

    # Merge system + user prompt if system_prompt provided
    if system_prompt:
        final_prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"
    else:
        final_prompt = user_prompt
    
    with mlflow.start_run():
        # Log input parameters
        mlflow.log_param("system_prompt", system_prompt or "none")
        mlflow.log_param("user_prompt", user_prompt)
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        # Track timing
        start_time = time.time()
        
        try:
            # Make request to llama.cpp server
            response = requests.post(
                f"{LLAMA_SERVER_URL}/completion",
                json={
                    "prompt": final_prompt,
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
            token_count = len(completion_text.split())  # crude token estimate
            
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
    # Plain prompt
    print("\n--- Normal Prompt Run ---")
    response1 = query_llama("Write a simple React button component with Tailwind CSS. Give me code so that i can paste into editor")
    if response1:
        print("Response:", response1[:300] + "..." if len(response1) > 300 else response1)

    # System prompt enforcing pure JSX
    print("\n--- System Prompt Run: JSX Only ---")
    response2 = query_llama(
        "Write a simple React button component with Tailwind CSS.Give me code so that i can paste into editor",
        system_prompt="You are a code generator. Always output only valid JSX code with no explanations.",
        model_params={"temperature": 0.2, "max_tokens": 300, "top_p": 0.9}
    )
    if response2:
        print("Response:", response2[:300] + "..." if len(response2) > 300 else response2)

    # System prompt with inline comments
    print("\n--- System Prompt Run: Senior Dev ---")
    response3 = query_llama(
        "Write a simple React button component with Tailwind CSS. Give me code so that i can paste into editor",
        system_prompt="You are a senior-level software engineer specializing in React and full-stack development. Your job is to produce accurate, production-quality React code for components and systems of any complexity",
        model_params={"temperature": 0.4, "max_tokens": 400, "top_p": 0.95}
    )
    if response3:
        print("Response:", response3[:300] + "..." if len(response3) > 300 else response3)

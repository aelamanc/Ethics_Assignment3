"""
DeepSeek-R1 API Evaluation Script for L4 Affective Awareness Indicators
Course: AI Safety Evaluation Assignment
Purpose: Generate model responses for 62 evaluation prompts (via DeepSeek API)
Model: DeepSeek-R1 (reasoning model with CoT capabilities)
"""

import pandas as pd
import requests
import json
from tqdm import tqdm
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Configuration
MODEL_NAME = "deepseek-reasoner"  # DeepSeek-R1's API name
API_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_CSV = "prompts_master.csv"
OUTPUT_JSON = f"deepseek_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
TEMPERATURE = 0.7
MAX_TOKENS = 1012
TOP_P = 0.9

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

def setup_api():
    """Initialize DeepSeek API"""
    if API_KEY is None:
        raise ValueError(
            "DEEPSEEK_API_KEY not found. Set it with: export DEEPSEEK_API_KEY='your_key'\n"
            "Get your key at: https://platform.deepseek.com/api_keys"
        )
    
    # Test API connection
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=test_data)
        if response.status_code == 200:
            print(f"✓ DeepSeek API connected successfully")
            print(f"✓ Using model: {MODEL_NAME}\n")
            return headers
        else:
            raise ValueError(f"API test failed: {response.status_code} - {response.text}")
    except Exception as e:
        raise ValueError(f"API connection error: {e}")

def generate_response(headers, user_text, retry_count=3):
    """
    Generate a single response from DeepSeek with retry logic
    """
    for attempt in range(retry_count):
        try:
            data = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": user_text}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "top_p": TOP_P,
            }
            
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # DeepSeek-R1 includes reasoning in response
                # Extract the actual response text
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]
                    
                    # Check for reasoning_content (CoT)
                    if "reasoning_content" in message:
                        # We want the final answer, not the reasoning process
                        content = message.get("content", "")
                    else:
                        content = message.get("content", "")
                    
                    return content.strip()
                else:
                    return "ERROR: No response content"
            
            elif response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                if attempt < retry_count - 1:
                    print(f"\n⚠ Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: Rate limit exceeded after {retry_count} attempts"
            
            else:
                error_msg = response.text
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"ERROR: API call failed - {response.status_code}: {error_msg[:100]}"
        
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                print(f"\n⚠ Timeout, retrying...")
                time.sleep(2)
            else:
                return "ERROR: Request timeout after retries"
        
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                return f"ERROR: {str(e)[:100]}"
    
    return "ERROR: Max retries exceeded"

def run_evaluation():
    """Main evaluation loop"""
    # Load prompts
    print(f"Loading prompts from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} prompts across {df['l4_indicator'].nunique()} L4 indicators")
    
    # Setup API
    print(f"\nInitializing DeepSeek API...")
    headers = setup_api()
    
    # Store results
    results = []
    error_count = 0
    
    # Generate responses
    print(f"Generating responses (temperature={TEMPERATURE})...")
    print(f"Note: DeepSeek-R1 may take longer per prompt due to reasoning process\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        prompt_id = row['id']
        l4_indicator = row['l4_indicator']
        user_text = row['user_text']
        
        response = generate_response(headers, user_text)
        
        # Track errors
        if response.startswith("ERROR"):
            error_count += 1
        
        result = {
            'id': prompt_id,
            'l4_indicator': l4_indicator,
            'user_text': user_text,
            'model_response': response,
            'model': MODEL_NAME,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_TOKENS,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        # Rate limiting (DeepSeek: be conservative)
        time.sleep(1.0)  # 1 second between requests
    
    # Save results
    print(f"\n\nSaving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Evaluation complete. Results saved to {OUTPUT_JSON}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts: {len(results)}")
    print(f"Model used: {MODEL_NAME}")
    print(f"Successfully generated: {len(results) - error_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print(f"\n⚠ {error_count} prompts resulted in errors")
        print("Check the JSON file for ERROR messages")
    
    print(f"\nL4 indicators: {df['l4_indicator'].nunique()}")
    for indicator in df['l4_indicator'].unique():
        total = len(df[df['l4_indicator'] == indicator])
        errors_in_indicator = sum(1 for r in results 
                                  if r['l4_indicator'] == indicator and 'ERROR' in r['model_response'])
        print(f"  - {indicator}: {total} prompts ({errors_in_indicator} errors)")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_evaluation()
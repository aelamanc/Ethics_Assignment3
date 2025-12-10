"""
Gemini Evaluation Script for L4 Affective Awareness Indicators
Course: AI Safety Evaluation Assignment
Purpose: Generate model responses for 62 evaluation prompts (via Google AI API)
Note: Explicit permission received from Google for prompt-based testing
"""

import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv


# Configuration
MODEL_NAME = "gemini-2.5-flash"  
INPUT_CSV = "prompts_master.csv"
OUTPUT_JSON = f"gemini_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024
TOP_P = 0.9

# API Key (set as environment variable: export GOOGLE_API_KEY="your_key")
load_dotenv() 
API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_gemini():
    """Initialize Gemini API"""
    if API_KEY is None:
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it with: export GOOGLE_API_KEY='your_key'"
        )
    
    genai.configure(api_key=API_KEY)
    
    # Configure generation parameters
    generation_config = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config
    )
    
    return model

def generate_response(model, user_text, retry_count=3):
    """
    Generate a single response from Gemini with retry logic
    """
    for attempt in range(retry_count):
        try:
            response = model.generate_content(user_text)
            
            # Check for safety blocks
            if response.prompt_feedback.block_reason:
                return f"BLOCKED: {response.prompt_feedback.block_reason}"
            
            return response.text.strip()
            
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nRetry {attempt + 1}/{retry_count} after error: {e}")
                time.sleep(wait_time)
            else:
                return f"ERROR after {retry_count} attempts: {str(e)}"
    
    return "ERROR: Max retries exceeded"

def run_evaluation():
    """Main evaluation loop"""
    # Load prompts
    print(f"Loading prompts from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} prompts across {df['l4_indicator'].nunique()} L4 indicators")
    
    # Setup Gemini
    print(f"\nInitializing Gemini ({MODEL_NAME})...")
    model = setup_gemini()
    print("✓ Gemini API configured")
    
    # Store results
    results = []
    
    # Generate responses
    print(f"\nGenerating responses (temperature={TEMPERATURE})...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt_id = row['id']
        l4_indicator = row['l4_indicator']
        user_text = row['user_text']
        
        response = generate_response(model, user_text)
        
        result = {
            'id': prompt_id,
            'l4_indicator': l4_indicator,
            'user_text': user_text,
            'model_response': response,
            'model': MODEL_NAME,
            'temperature': TEMPERATURE,
            'max_tokens': MAX_OUTPUT_TOKENS,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        # Rate limiting (Gemini free tier: 15 RPM)
        time.sleep(0.5)
    
    # Save results
    print(f"\nSaving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Evaluation complete. Results saved to {OUTPUT_JSON}")
    
    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Total prompts: {len(results)}")
    print(f"L4 indicators: {df['l4_indicator'].nunique()}")
    for indicator in df['l4_indicator'].unique():
        count = len(df[df['l4_indicator'] == indicator])
        print(f"  - {indicator}: {count} prompts")
    
    # Check for blocks/errors
    blocked = sum(1 for r in results if 'BLOCKED' in r['model_response'])
    errors = sum(1 for r in results if 'ERROR' in r['model_response'])
    if blocked > 0:
        print(f"\n⚠ {blocked} prompts were blocked by safety filters")
    if errors > 0:
        print(f"⚠ {errors} prompts resulted in errors")

if __name__ == "__main__":
    run_evaluation()

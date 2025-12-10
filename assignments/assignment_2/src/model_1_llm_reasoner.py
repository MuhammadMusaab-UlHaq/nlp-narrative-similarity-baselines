"""
Assignment 2: LLM Reasoner Baseline (Chain-of-Thought)
Author: Muhammad Musaab ul Haq

Objective: Use GPT-4o-mini (From poe.com) to perform Chain-of-Thought reasoning
before making a prediction on narrative similarity.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# --- Configuration ---
# Define paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # assignments/assignment_2/src -> assignments/assignment_2 -> assignments -> root
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dev_track_a.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "assignments" / "assignment_2" / "output" / "musaab_ul_haq"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup OpenAI Client
client = None
MODEL_NAME = "gpt-4o-mini" # Default

def setup_client(provider, api_key=None, model_name=None):
    global client, MODEL_NAME
    
    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API Key not found. Set GEMINI_API_KEY environment variable.")
        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        MODEL_NAME = model_name or "gemini-2.0-flash"
        print(f"Using Gemini API with model {MODEL_NAME}")
        
    elif provider == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
             raise ValueError("OpenRouter API Key not found")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        MODEL_NAME = model_name or "openai/gpt-4o-mini"
        print(f"Using OpenRouter API with model {MODEL_NAME}")

    elif provider == "poe":
        key = api_key or os.getenv("POE_API_KEY")
        if not key:
             raise ValueError("Poe API Key not found. Set POE_API_KEY environment variable.")
        client = OpenAI(
            api_key=key,
            base_url="https://api.poe.com/v1"
        )
        MODEL_NAME = model_name or "gpt-4o-mini"
        print(f"Using Poe API with model {MODEL_NAME}")

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
             raise ValueError("OpenAI API Key not found")
        client = OpenAI(api_key=key)
        MODEL_NAME = model_name or "gpt-4o-mini"
        print(f"Using OpenAI API with model {MODEL_NAME}")

    elif provider == "mistral":
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
             raise ValueError("Mistral API Key not found. Set MISTRAL_API_KEY environment variable.")
        client = OpenAI(
            api_key=key,
            base_url="https://api.mistral.ai/v1"
        )
        MODEL_NAME = model_name or "mistral-small-latest"
        print(f"Using Mistral API with model {MODEL_NAME}")

    else:
        raise ValueError(f"Unknown provider: {provider}")

COT_PROMPT = """
You are an expert in narrative structure analysis. Your task is to identify which of two candidate stories (Story A or Story B) shares the most **structural similarity** with an Anchor Story.

**DEFINITION OF NARRATIVE STRUCTURE:**
Structure is NOT just "theme" (e.g., love, war). It is the specific combination of:
1. **Concrete Setting & Time**: (e.g., "19th Century France", "Modern High School", "Spaceship").
2. **Core Conflict Mechanism**: (e.g., "Courtroom Trial", "Heist", "Vampire Attack", "Political Election").
3. **Protagonist Archetype**: (e.g., "Detective", "Orphan", "Mad Scientist", "Soldier").
4. **Tone/Genre**: (e.g., "Satire", "Gothic Horror", "Romantic Comedy").

**CRITICAL INSTRUCTIONS:**
- **Prioritize CONCRETE details over abstract themes.**
    - *Example:* If Anchor is about a "Vampire", and Story A is about a "Blood-sucking Plant", and Story B is about a "Murderer", Story A is the match because the *mechanism* (draining blood) is structurally identical, even if B is also a "killer".
- **Watch out for GENRE mismatches.**
    - *Example:* A "Political Satire" (Comedy) is NOT structurally similar to a "Political Thriller" (Serious), even if both feature a President.
- **Watch out for SETTING mismatches.**
    - *Example:* A "Hospital Drama" is structurally different from a "Domestic Drama", even if both involve a married couple.

**ANALYSIS PROCESS:**
1. **Anchor Analysis**: List the Concrete Setting, Time Period, Core Conflict Mechanism, and Protagonist Archetype.
2. **Story A Analysis**: List the same 4 elements.
3. **Story B Analysis**: List the same 4 elements.
4. **Compare**:
   - Compare Anchor vs A on the 4 elements.
   - Compare Anchor vs B on the 4 elements.
   - Identify the "Dealbreakers" (e.g., mismatched Time Period, mismatched Genre).
   - Identify the "Strong Links" (e.g., shared specific plot device like "Time Travel" or "Kidnapping").

**DECISION LOGIC:**
- Select the story that matches the **most specific** structural elements.
- If one story matches the **Time Period** and **Setting** but another matches the **Theme**, prioritize **Time Period/Setting** + **Conflict Mechanism**.

**Output Format:**
You must output a valid JSON object:
{
    "reasoning": "Your step-by-step analysis...",
    "prediction": "A" or "B"
}
"""

def predict_with_cot(anchor, text_a, text_b):
    """
    Sends the stories to the LLM and retrieves the prediction and reasoning.
    """
    if not client:
        return False, "Client not initialized"

    user_content = f"Anchor Story:\n{anchor}\n\nStory A:\n{text_a}\n\nStory B:\n{text_b}"
    
    try:
        extra_headers = {}
        if "openrouter.ai" in str(client.base_url):
             extra_headers = {
                "HTTP-Referer": "https://github.com/MuhammadMusaab-UlHaq/ai_sem_proj_semeval-2026-task-4-baselines", 
                "X-Title": "SemEval 2026 Task 4 Baseline", 
            }

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": COT_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0, # Deterministic output
            extra_headers=extra_headers
        )
        content = response.choices[0].message.content
        
        # Sanitize content to remove markdown code blocks if present, specifically for POE, might not be needed for openai api.
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        result = json.loads(content)
        
        # Normalize prediction to boolean (True if A is closer)
        pred_label = result.get('prediction', '').strip().upper()
        is_a = (pred_label == 'A')
        
        return is_a, result.get('reasoning', 'No reasoning provided')
        
    except Exception as e:
        print(f"Error during API call: {e}")
        return None, f"Error: {str(e)}"

def process_row(row_data):
    """
    Helper function to process a single row for parallel execution.
    """
    index, row = row_data
    pred_is_a, reasoning = predict_with_cot(row['anchor_text'], row['text_a'], row['text_b'])
    
    truth_is_a = row['text_a_is_closer']
    
    is_correct = False
    if pred_is_a is not None:
        is_correct = (pred_is_a == truth_is_a)

    return {
        "index": index,
        "anchor_text": row['anchor_text'][:100] + "...", # Truncate for readability in JSON
        "prediction_is_a": pred_is_a,
        "truth_is_a": truth_is_a,
        "correct": is_correct,
        "reasoning": reasoning,
        "error": pred_is_a is None
    }

def main():
    parser = argparse.ArgumentParser(description="Run LLM CoT Baseline")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to run (for testing)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--provider", type=str, default="gemini", choices=["openai", "openrouter", "poe", "gemini", "mistral"], help="LLM Provider")
    parser.add_argument("--api_key", type=str, default=None, help="API Key (optional, overrides env vars)")
    parser.add_argument("--model", type=str, default=None, help="Model name (optional, overrides default)")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated list of indices to process (e.g., '1,2,3')")
    args = parser.parse_args()

    try:
        setup_client(args.provider, args.api_key, args.model)
    except Exception as e:
        print(f"Error setting up client: {e}")
        return

    if not client:
        print("Error: OpenAI client could not be initialized. Check your API key.")
        return

    print(f"Loading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_json(DATA_PATH, lines=True)
    
    if args.indices:
        indices_to_run = [int(i.strip()) for i in args.indices.split(',')]
        print(f"Filtering for indices: {indices_to_run}")
        # Ensure the dataframe has an 'index' column or use the index of the dataframe if it matches
        # The input jsonl usually doesn't have an explicit 'index' field in the json object unless added.
        # Assuming the 'index' in results refers to the line number (0-based).
        # Let's check if 'index' column exists, if not use the dataframe index.
        if 'index' in df.columns:
             df = df[df['index'].isin(indices_to_run)]
        else:
             df = df.iloc[indices_to_run]
    elif args.limit:
        print(f"Limiting run to first {args.limit} samples.")
        df = df.head(args.limit)
    
    print(f"Running LLM CoT Baseline on {len(df)} samples with {args.workers} workers...")
    
    results = []
    correct_count = 0
    processed_count = 0
    
    # Prepare data for parallel processing
    rows_to_process = list(df.iterrows())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_row, row): row for row in rows_to_process}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(rows_to_process)):
            try:
                result = future.result()
                results.append(result)
                
                if not result.get('error'):
                    processed_count += 1
                    if result['correct']:
                        correct_count += 1
            except Exception as e:
                print(f"Generated an exception: {e}")
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])

    # Calculate metrics
    if processed_count > 0:
        accuracy = correct_count / processed_count
    else:
        accuracy = 0.0
        
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct_count}/{processed_count})")
    
    # Save Results
    output_file = OUTPUT_DIR / "musaab_cot_results.jsonl"
    print(f"Saving detailed results to: {output_file}")
    
    out_df = pd.DataFrame(results)
    out_df.to_json(output_file, orient='records', lines=True)
    
    # Save a summary report
    summary_file = OUTPUT_DIR / "results_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"LLM Reasoner Baseline (GPT-4o-mini)\n")
        f.write(f"===================================\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Samples processed: {len(df)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Output file: {output_file}\n")

if __name__ == "__main__":
    main()

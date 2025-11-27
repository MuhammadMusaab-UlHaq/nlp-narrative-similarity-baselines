"""
Assignment 2: LLM Reasoner Baseline (Chain-of-Thought)
Author: Muhammad Musaab ul Haq

Objective: Use GPT-4o-mini to perform Chain-of-Thought reasoning
before making a prediction on narrative similarity.
"""

import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json

# Setup
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

COT_PROMPT = """
You are an expert in narrative theory. You are given an Anchor Story and two choices (Story A and Story B).
Your task is to determine which story is narratively similar to the Anchor.

Narrative similarity is defined by:
1. Abstract Theme (The core ideas/motives)
2. Course of Action (Sequence of events)
3. Outcome (The result/resolution)

Step 1: Analyze the Anchor Story's theme, action, and outcome.
Step 2: Analyze Story A's theme, action, and outcome.
Step 3: Analyze Story B's theme, action, and outcome.
Step 4: Compare A and B to the Anchor.
Step 5: Output your final decision as exactly 'A' or 'B'.

Return your response in JSON format:
{
    "reasoning": "YOUR STEP-BY-STEP REASONING HERE",
    "prediction": "A" or "B"
}
"""

def predict_with_cot(anchor, text_a, text_b):
    user_content = f"Anchor: {anchor}\n\nStory A: {text_a}\n\nStory B: {text_b}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": COT_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        return result['prediction'] == 'A', result['reasoning']
    except Exception as e:
        print(f"Error: {e}")
        return False, "Error"

def main():
    # Load Data
    data_path = "../../data/raw/dev_track_a.jsonl"
    df = pd.read_json(data_path, lines=True)
    
    print(f"Running LLM CoT Baseline on {len(df)} samples...")
    
    # Run Inference (Small sample for testing, remove .head() for full run)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred_is_a, reasoning = predict_with_cot(row['anchor_text'], row['text_a'], row['text_b'])
        results.append({
            "prediction_is_a": pred_is_a,
            "truth_is_a": row['text_a_is_closer'],
            "reasoning": reasoning
        })
    
    # Save Results
    out_df = pd.DataFrame(results)
    accuracy = (out_df['prediction_is_a'] == out_df['truth_is_a']).mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    os.makedirs("../output", exist_ok=True)
    out_df.to_json("../output/musaab_cot_results.jsonl", orient='records', lines=True)

if __name__ == "__main__":
    main()

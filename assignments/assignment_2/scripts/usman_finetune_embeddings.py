"""
Fine-Tuned Embedding Baseline using Contrastive Learning
Author: Usman Amjad
Assignment 2 - SemEval 2026 Task 4

This script implements a fine-tuned SentenceTransformer model using contrastive learning
on synthetic data to improve semantic similarity detection for narrative texts.
"""

import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers.util import cos_sim
import numpy as np
from datetime import datetime

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "assignments" / "assignment_2" / "output" / "usman_amjad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_contrastive_data(jsonl_path):
    """
    Load synthetic data for contrastive learning.
    
    Args:
        jsonl_path: Path to the JSONL file containing triplets
        
    Returns:
        List of (anchor, positive, negative) tuples
    """
    triplets = []
    
    print(f"Loading training data from: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            
            anchor = record.get('anchor_story')
            similar = record.get('similar_story')
            dissimilar = record.get('dissimilar_story')
            
            # Skip records with missing data
            if anchor and similar and dissimilar:
                triplets.append((anchor, similar, dissimilar))
    
    print(f"✓ Loaded {len(triplets)} training triplets")
    return triplets


def prepare_training_examples(triplets):
    """
    Convert triplets to InputExample format for sentence-transformers.
    
    For contrastive learning, we create pairs with similarity labels:
    - (anchor, similar) with label 1.0 (high similarity)
    - (anchor, dissimilar) with label 0.0 (low similarity)
    
    Args:
        triplets: List of (anchor, positive, negative) tuples
        
    Returns:
        List of InputExample objects
    """
    examples = []
    
    for anchor, positive, negative in triplets:
        # Create positive pair (anchor, similar) with label 1
        examples.append(InputExample(texts=[anchor, positive], label=1.0))
        
        # Create negative pair (anchor, dissimilar) with label 0
        examples.append(InputExample(texts=[anchor, negative], label=0.0))
    
    print(f"✓ Created {len(examples)} training examples (positive + negative pairs)")
    return examples


def finetune_model(
    model_name="all-MiniLM-L6-v2",
    train_examples=None,
    batch_size=16,
    epochs=4,
    warmup_steps=100,
    output_path=None
):
    """
    Fine-tune a SentenceTransformer model using contrastive loss.
    
    Args:
        model_name: Name of the pre-trained model to fine-tune
        train_examples: List of InputExample objects
        batch_size: Training batch size
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        output_path: Path to save the fine-tuned model
        
    Returns:
        Fine-tuned SentenceTransformer model
    """
    print(f"\nLoading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size
    )
    
    # Use CosineSimilarityLoss for contrastive learning
    # This loss expects pairs with similarity labels (0 or 1)
    train_loss = losses.CosineSimilarityLoss(model)
    
    print(f"\n{'='*60}")
    print(f"Starting Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Base model:         {model_name}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Epochs:             {epochs}")
    print(f"  Training examples:  {len(train_examples)}")
    print(f"  Warmup steps:       {warmup_steps}")
    print(f"{'='*60}\n")
    
    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        show_progress_bar=True,
        save_best_model=True
    )
    
    print(f"\n✓ Model saved to: {output_path}")
    return model


def evaluate_model(model, dev_data_path):
    """
    Evaluate model on Track A dev set using cosine similarity approach.
    
    For each instance, we:
    1. Encode the anchor, text_a, and text_b
    2. Compute cosine similarity between anchor and each text
    3. Predict that text_a is closer if sim(anchor, text_a) > sim(anchor, text_b)
    4. Compare with ground truth label
    
    Args:
        model: SentenceTransformer model
        dev_data_path: Path to dev set JSONL file
        
    Returns:
        Accuracy score
    """
    print(f"\nEvaluating on: {dev_data_path}")
    
    # Load dev data
    dev_data = []
    with open(dev_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dev_data.append(json.loads(line))
    
    print(f"✓ Loaded {len(dev_data)} dev examples")
    
    # Collect all unique texts for efficient encoding
    all_texts = []
    for record in dev_data:
        all_texts.extend([record['anchor_text'], record['text_a'], record['text_b']])
    
    unique_texts = list(set(all_texts))
    print(f"✓ Encoding {len(unique_texts)} unique texts...")
    
    # Generate embeddings
    embeddings = model.encode(unique_texts, show_progress_bar=True, convert_to_tensor=True)
    embedding_lookup = dict(zip(unique_texts, embeddings))
    
    # Evaluate
    correct = 0
    total = len(dev_data)
    
    print(f"✓ Computing predictions...")
    
    for record in dev_data:
        anchor_emb = embedding_lookup[record['anchor_text']]
        text_a_emb = embedding_lookup[record['text_a']]
        text_b_emb = embedding_lookup[record['text_b']]
        
        # Calculate cosine similarities
        sim_a = cos_sim(anchor_emb, text_a_emb).item()
        sim_b = cos_sim(anchor_emb, text_b_emb).item()
        
        # Predict based on higher similarity
        prediction = sim_a > sim_b
        
        # Check correctness
        if prediction == record['text_a_is_closer']:
            correct += 1
    
    accuracy = correct / total
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*60}\n")
    
    return accuracy


def main():
    """
    Main training and evaluation pipeline.
    """
    print("\n" + "="*60)
    print("Fine-Tuned Embedding Baseline")
    print("Author: Usman Amjad")
    print("Assignment 2 - SemEval 2026 Task 4")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    
    # Paths
    synthetic_data = DATA_DIR / "processed" / "synthetic_data_for_contrastive_learning.jsonl"
    dev_data = DATA_DIR / "raw" / "dev_track_a.jsonl"
    model_output = OUTPUT_DIR / "finetuned_model"
    
    # Hyperparameters
    MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 16
    EPOCHS = 4
    WARMUP_STEPS = 100
    
    # Step 1: Load training data
    print("\n[1/4] Loading Training Data")
    print("-" * 60)
    triplets = load_contrastive_data(synthetic_data)
    train_examples = prepare_training_examples(triplets)
    
    # Step 2: Fine-tune model
    print("\n[2/4] Fine-Tuning Model")
    print("-" * 60)
    model = finetune_model(
        model_name=MODEL_NAME,
        train_examples=train_examples,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=model_output
    )
    
    # Step 3: Evaluate on dev set
    print("\n[3/4] Evaluating on Dev Set")
    print("-" * 60)
    accuracy = evaluate_model(model, dev_data)
    
    # Step 4: Save results
    print("\n[4/4] Saving Results")
    print("-" * 60)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        "model": MODEL_NAME,
        "training_triplets": len(triplets),
        "training_samples": len(train_examples),
        "dev_accuracy": accuracy,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "training_duration_seconds": duration,
        "timestamp": start_time.isoformat()
    }
    
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    print(f"✓ Model saved to: {model_output}")
    print(f"✓ Training duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    print("\n" + "="*60)
    print("DONE! Fine-tuning and evaluation completed successfully.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

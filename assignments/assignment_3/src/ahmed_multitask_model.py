import json
import logging
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. SETUP & LOGGING
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_data_for_multitask(filepath):
    """
    Parses the JSONL data and creates two separate datasets for Multi-Task Learning.
    
    Task A: Semantic Ranking (Triplet Data)
    Task B: Binary Classification (Pair Data + Label)
    """
    ranking_samples = []        # Data for Head 1
    classification_samples = [] # Data for Head 2
    
    filepath = os.path.normpath(filepath)
    print(f"Loading data from: {filepath}")
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # Extract fields
                anchor = item.get('anchor_story')
                positive = item.get('similar_story')
                negative = item.get('dissimilar_story')
                
                if not all([anchor, positive, negative]):
                    continue
                
                # --- TASK 1 SETUP: Ranking ---
                # Format: [Anchor, Positive, Negative]
                # The model learns to pull Positive closer and push Negative away.
                ranking_samples.append(InputExample(texts=[anchor, positive, negative]))

                # --- TASK 2 SETUP: Classification ---
                # Format: [Text1, Text2], Label
                # We create explicit "Match" (1) and "No Match" (0) examples.
                
                # Positive Pair -> Label 1
                classification_samples.append(InputExample(texts=[anchor, positive], label=1))
                
                # Negative Pair -> Label 0
                classification_samples.append(InputExample(texts=[anchor, negative], label=0))

        print(f"Data Loaded: {len(ranking_samples)} ranking triplets, {len(classification_samples)} classification pairs.")
        return ranking_samples, classification_samples
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def train_multitask_model():
    # --- CONFIGURATION ---
    model_name = 'all-MiniLM-L6-v2'
    train_batch_size = 16
    num_epochs = 4
    
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'output', 'multitask_model_v1')
    
    # Robust Path Finding (Goes up 3 levels to find 'data' folder)
    # 1. Try relative path
    # To this (combines both datasets):
    train_path_original = os.path.join(script_dir, '..', '..', '..', 'data', 'processed', 'combined_synthetic_for_training.jsonl') # Muhammad Musaab Ul Haq: changed the paths for the convenience of Usman Amjad. 
    train_path_augmented = os.path.join(script_dir, '..', '..', '..', 'data', 'processed', 'augmented_synthetic_500.jsonl')
    
    # 2. Fallback to the absolute path if relative fails (based on your previous logs)
    if not os.path.exists(os.path.normpath(train_path)):
        train_path = r"c:\Users\ahmed\OneDrive\Desktop\NUST\3 SEM\AI\PROJECT_AI\ai_sem_proj_semeval-2026-task-4-baselines\data\processed\combined_synthetic_for_training.jsonl"

    # --- 1. INITIALIZE SHARED MODEL ---
    print(f"Loading Shared Transformer: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # --- 2. PREPARE MULTI-TASK DATA ---
    train_ranking, train_classif = load_data_for_multitask(train_path)
    
    if not train_ranking:
        print("Data load failed. Exiting.")
        return
    
    # Create distinct DataLoaders for each task
    loader_ranking = DataLoader(train_ranking, shuffle=True, batch_size=train_batch_size)
    loader_classif = DataLoader(train_classif, shuffle=True, batch_size=train_batch_size)
    
    # --- 3. DEFINE MULTI-TASK HEADS (LOSS FUNCTIONS) ---
    
    # HEAD 1: Semantic Search Head
    # Loss: MultipleNegativesRankingLoss
    # Goal: Optimize vector space (Sim(A, P) > Sim(A, N))
    train_loss_ranking = losses.MultipleNegativesRankingLoss(model=model)
    
    # HEAD 2: Classification Head
    # Loss: SoftmaxLoss
    # Goal: Explicitly classify pairs as 0 or 1.
    # Note: This adds a linear layer on top of the model embeddings with output dim = 2
    train_loss_classif = losses.SoftmaxLoss(
        model=model, 
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
        num_labels=2 
    )
    
    # --- 4. TRAIN WITH MULTI-TASK OBJECTIVE ---
    print("\nStarting Multi-Task Training...")
    print("Strategy: Alternating batches between Ranking Loss and Classification Loss")
    
    model.fit(
        train_objectives=[
            (loader_ranking, train_loss_ranking), # Task 1
            (loader_classif, train_loss_classif)  # Task 2
        ],
        epochs=num_epochs,
        warmup_steps=100,
        output_path=model_save_path,
        show_progress_bar=True
    )
    
    print(f"Multi-Task Training complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train_multitask_model()
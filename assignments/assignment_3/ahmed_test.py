import json
import logging
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. SETUP & LOGGING
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_data_for_multitask(filepath, max_samples=200):
    """
    Parses the JSONL data and creates two separate datasets for Multi-Task Learning.
    
    Args:
        max_samples (int): LIMITS data loading to first N items for speed testing.
    """
    ranking_samples = []        # Data for Head 1
    classification_samples = [] # Data for Head 2
    
    filepath = os.path.normpath(filepath)
    print(f"Loading data from: {filepath}")
    print(f"⚠️ SPEED MODE: Limiting to first {max_samples} samples only.")
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                # --- SPEED LIMIT CHECK ---
                if count >= max_samples:
                    break
                
                if not line.strip(): continue
                item = json.loads(line)
                
                # Extract fields
                anchor = item.get('anchor_story')
                positive = item.get('similar_story')
                negative = item.get('dissimilar_story')
                
                if not all([anchor, positive, negative]):
                    continue
                
                # --- TASK 1 SETUP: Ranking ---
                ranking_samples.append(InputExample(texts=[anchor, positive, negative]))

                # --- TASK 2 SETUP: Classification ---
                classification_samples.append(InputExample(texts=[anchor, positive], label=1))
                classification_samples.append(InputExample(texts=[anchor, negative], label=0))
                
                count += 1

        print(f"Data Loaded: {len(ranking_samples)} ranking triplets, {len(classification_samples)} classification pairs.")
        return ranking_samples, classification_samples
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def train_multitask_model():
    # --- CONFIGURATION (OPTIMIZED FOR SPEED) ---
    model_name = 'all-MiniLM-L6-v2'
    train_batch_size = 16
    
    # ⚡ FAST SETTINGS ⚡
    num_epochs = 1          # Reduced from 4 to 1
    max_samples_to_load = 200 # Only load 200 items
    
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'output', 'multitask_model_TEST_RUN')
    
    # Robust Path Finding
    train_path = os.path.join(script_dir, '..', '..', '..', 'data', 'processed', 'combined_synthetic_for_training.jsonl')
    if not os.path.exists(os.path.normpath(train_path)):
        train_path = r"c:\Users\ahmed\OneDrive\Desktop\NUST\3 SEM\AI\PROJECT_AI\ai_sem_proj_semeval-2026-task-4-baselines\data\processed\combined_synthetic_for_training.jsonl"

    # --- 1. INITIALIZE SHARED MODEL ---
    print(f"Loading Shared Transformer: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # --- 2. PREPARE MULTI-TASK DATA (WITH LIMIT) ---
    train_ranking, train_classif = load_data_for_multitask(train_path, max_samples=max_samples_to_load)
    
    if not train_ranking:
        print("Data load failed. Exiting.")
        return
    
    # Create distinct DataLoaders
    loader_ranking = DataLoader(train_ranking, shuffle=True, batch_size=train_batch_size)
    loader_classif = DataLoader(train_classif, shuffle=True, batch_size=train_batch_size)
    
    # --- 3. DEFINE MULTI-TASK HEADS ---
    train_loss_ranking = losses.MultipleNegativesRankingLoss(model=model)
    
    train_loss_classif = losses.SoftmaxLoss(
        model=model, 
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
        num_labels=2 
    )
    
    # --- 4. TRAIN WITH MULTI-TASK OBJECTIVE ---
    print("\nStarting FAST Multi-Task Training...")
    
    model.fit(
        train_objectives=[
            (loader_ranking, train_loss_ranking),
            (loader_classif, train_loss_classif)
        ],
        epochs=num_epochs,
        warmup_steps=10, # Reduced warmup for small data
        output_path=model_save_path,
        show_progress_bar=True
    )
    
    print(f"Test Run Complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    train_multitask_model()
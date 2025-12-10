import json
import logging
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_test_data(filepath):
    """
    Load test data from JSONL file.
    Supports Track A/B format: anchor_text, text_a, text_b, text_a_is_closer
    """
    test_pairs = []
    
    filepath = os.path.normpath(filepath)
    logging.info(f"Loading test data from: {filepath}")
    
    if not os.path.exists(filepath):
        logging.error(f"Test file not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Track A/B format: anchor_text, text_a, text_b, text_a_is_closer
            if 'anchor_text' in item and 'text_a' in item and 'text_b' in item:
                anchor = item['anchor_text']
                text_a = item['text_a']
                text_b = item['text_b']
                text_a_is_closer = item.get('text_a_is_closer', True)
                
                # Create pairs based on which text is closer
                if text_a_is_closer:
                    # text_a is similar (label=1), text_b is dissimilar (label=0)
                    test_pairs.append({
                        'text1': anchor,
                        'text2': text_a,
                        'label': 1
                    })
                    test_pairs.append({
                        'text1': anchor,
                        'text2': text_b,
                        'label': 0
                    })
                else:
                    # text_b is similar (label=1), text_a is dissimilar (label=0)
                    test_pairs.append({
                        'text1': anchor,
                        'text2': text_b,
                        'label': 1
                    })
                    test_pairs.append({
                        'text1': anchor,
                        'text2': text_a,
                        'label': 0
                    })
            
            # Training data format: anchor_story, similar_story, dissimilar_story
            elif 'anchor_story' in item:
                anchor = item['anchor_story']
                positive = item['similar_story']
                negative = item['dissimilar_story']
                
                test_pairs.append({
                    'text1': anchor,
                    'text2': positive,
                    'label': 1
                })
                test_pairs.append({
                    'text1': anchor,
                    'text2': negative,
                    'label': 0
                })
    
    logging.info(f"Loaded {len(test_pairs)} test pairs")
    return test_pairs

def evaluate_model(model_path, test_data_path, output_dir):
    """
    Evaluate the trained model on test data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Load trained model
    logging.info(f"Loading model from: {model_path}")
    model = SentenceTransformer(model_path, device=device)
    
    # Load test data
    test_pairs = load_test_data(test_data_path)
    
    if not test_pairs:
        logging.error("No test data loaded. Exiting.")
        return None
    
    # Extract texts and labels
    texts1 = [pair['text1'] for pair in test_pairs]
    texts2 = [pair['text2'] for pair in test_pairs]
    true_labels = np.array([pair['label'] for pair in test_pairs])
    
    # Compute embeddings
    logging.info("Computing embeddings...")
    embeddings1 = model.encode(texts1, show_progress_bar=True, convert_to_tensor=True, device=device)
    embeddings2 = model.encode(texts2, show_progress_bar=True, convert_to_tensor=True, device=device)
    
    # Compute cosine similarities
    logging.info("Computing similarities...")
    similarities = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()
    
    # Find optimal threshold using F1 score
    logging.info("Finding optimal threshold...")
    thresholds = np.arange(0.1, 1.0, 0.02)
    best_threshold = 0.5
    best_f1 = 0
    threshold_metrics = []
    
    for threshold in thresholds:
        pred_labels = (similarities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        threshold_metrics.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logging.info(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Final predictions with best threshold
    pred_labels = (similarities >= best_threshold).astype(int)
    
    # Calculate final metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary'
    )
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Calculate additional statistics
    positive_sims = similarities[true_labels == 1]
    negative_sims = similarities[true_labels == 0]
    
    # --- SAVE RESULTS ---
    results = {
        'model_path': model_path,
        'test_data': test_data_path,
        'device': device,
        'num_test_pairs': len(test_pairs),
        'num_positive': int(np.sum(true_labels == 1)),
        'num_negative': int(np.sum(true_labels == 0)),
        'best_threshold': float(best_threshold),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'similarity_stats': {
            'positive_pairs': {
                'mean': float(np.mean(positive_sims)),
                'std': float(np.std(positive_sims)),
                'min': float(np.min(positive_sims)),
                'max': float(np.max(positive_sims))
            },
            'negative_pairs': {
                'mean': float(np.mean(negative_sims)),
                'std': float(np.std(negative_sims)),
                'min': float(np.min(negative_sims)),
                'max': float(np.max(negative_sims))
            }
        },
        'confusion_matrix': {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
    }
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    
    # --- PLOT 1: Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Dissimilar (0)', 'Similar (1)'],
                yticklabels=['Dissimilar (0)', 'Similar (1)'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_plot = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot, dpi=150)
    logging.info(f"Confusion matrix saved to: {cm_plot}")
    plt.close()
    
    # --- PLOT 2: Similarity Distribution ---
    plt.figure(figsize=(12, 6))
    
    bins = 50
    plt.hist(negative_sims, bins=bins, alpha=0.6, label='Dissimilar Pairs', color='red', edgecolor='black')
    plt.hist(positive_sims, bins=bins, alpha=0.6, label='Similar Pairs', color='green', edgecolor='black')
    plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Best Threshold: {best_threshold:.3f}')
    
    plt.xlabel('Cosine Similarity Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Similarity Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    dist_plot = os.path.join(output_dir, 'similarity_distribution.png')
    plt.savefig(dist_plot, dpi=150)
    logging.info(f"Similarity distribution saved to: {dist_plot}")
    plt.close()
    
    # --- PLOT 3: Threshold vs Metrics ---
    plt.figure(figsize=(12, 6))
    
    thresholds_arr = [m['threshold'] for m in threshold_metrics]
    f1_scores = [m['f1'] for m in threshold_metrics]
    precisions = [m['precision'] for m in threshold_metrics]
    recalls = [m['recall'] for m in threshold_metrics]
    
    plt.plot(thresholds_arr, f1_scores, label='F1 Score', linewidth=2, marker='o', markersize=3)
    plt.plot(thresholds_arr, precisions, label='Precision', linewidth=2, marker='s', markersize=3)
    plt.plot(thresholds_arr, recalls, label='Recall', linewidth=2, marker='^', markersize=3)
    plt.axvline(best_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Best Threshold: {best_threshold:.3f}')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    threshold_plot = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(threshold_plot, dpi=150)
    logging.info(f"Threshold analysis saved to: {threshold_plot}")
    plt.close()
    
    # --- PRINT SUMMARY ---
    print("\n" + "="*70)
    print("EVALUATION RESULTS".center(70))
    print("="*70)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Test Data: {os.path.basename(test_data_path)}")
    print(f"Device: {device}")
    print("-"*70)
    print(f"Test Samples: {len(test_pairs):,}")
    print(f"  - Similar Pairs (Label 1): {results['num_positive']:,}")
    print(f"  - Dissimilar Pairs (Label 0): {results['num_negative']:,}")
    print("-"*70)
    print(f"Best Threshold: {best_threshold:.4f}")
    print("-"*70)
    print("METRICS:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("-"*70)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives:  {cm[0, 0]:,}")
    print(f"  False Positives: {cm[0, 1]:,}")
    print(f"  False Negatives: {cm[1, 0]:,}")
    print(f"  True Positives:  {cm[1, 1]:,}")
    print("-"*70)
    print("SIMILARITY STATISTICS:")
    print(f"  Similar Pairs:    μ={np.mean(positive_sims):.4f}, σ={np.std(positive_sims):.4f}")
    print(f"  Dissimilar Pairs: μ={np.mean(negative_sims):.4f}, σ={np.std(negative_sims):.4f}")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("="*70 + "\n")
    
    return results

if __name__ == "__main__":
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model path (your trained model)
    model_path = os.path.join(script_dir, 'experiments', 'EXP_001_Combined_Data')
    
    # Test datasets (choose which one to evaluate)
    test_datasets = {
        'dev_track_a': os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'dev_track_a.jsonl'),
        'dev_track_b': os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'dev_track_b.jsonl'),
        'sample_track_a': os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'sample_track_a.jsonl'),
        'sample_track_b': os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'sample_track_b.jsonl'),
    }
    
    # Evaluate on all test sets
    for dataset_name, test_path in test_datasets.items():
        test_path_norm = os.path.normpath(test_path)
        
        if not os.path.exists(test_path_norm):
            logging.warning(f"Skipping {dataset_name}: File not found")
            continue
        
        logging.info(f"\n{'='*70}")
        logging.info(f"EVALUATING ON: {dataset_name}")
        logging.info(f"{'='*70}\n")
        
        output_dir = os.path.join(script_dir, 'experiments', 'EXP_001_Combined_Data', 'evaluation', dataset_name)
        
        evaluate_model(model_path, test_path_norm, output_dir)
        
        print("\n" + "="*70 + "\n")

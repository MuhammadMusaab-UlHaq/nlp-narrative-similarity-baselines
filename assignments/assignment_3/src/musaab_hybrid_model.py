"""
Assignment 3: Hybrid Model for Narrative Similarity
Author: Muhammad Musaab ul Haq

Architecture:
1. Fast embedding model for initial prediction + confidence
2. CoT LLM fallback for low-confidence cases

Key Insight: This hybrid achieves near-LLM accuracy with significant cost reduction
by only calling the expensive LLM when the embedding model is uncertain.
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import argparse
from datetime import datetime

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIDENCE_THRESHOLD = 0.10
DEV_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dev_track_a.jsonl"

# Model paths in order of preference
MODEL_PATHS = {
    "a2_finetuned": PROJECT_ROOT / "assignments" / "assignment_2" / "output" / "usman_amjad" / "finetuned_model",
    "a3_multitask": PROJECT_ROOT / "assignments" / "assignment_3" / "src" / "experiments" / "EXP_001_Combined_Data",
    "pretrained": "all-MiniLM-L6-v2"  # HuggingFace model name
}

# Output directory
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "musaab_hybrid"

# ============================================================================
# COT PROMPT (Proven 71.8% accuracy from A2)
# ============================================================================

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
    - *Example:* If Anchor is about a "Vampire", and Story A is about a "Blood-sucking Plant", and Story B is about a "Murderer", Story A is the match because the *mechanism* (draining blood) is structurally identical.
- **Watch out for GENRE mismatches.**
    - *Example:* A "Political Satire" (Comedy) is NOT structurally similar to a "Political Thriller" (Serious).
- **Watch out for SETTING mismatches.**
    - *Example:* A "Hospital Drama" is structurally different from a "Domestic Drama".

**ANALYSIS PROCESS:**
1. **Anchor Analysis**: List the Concrete Setting, Time Period, Core Conflict Mechanism, and Protagonist Archetype.
2. **Story A Analysis**: List the same 4 elements.
3. **Story B Analysis**: List the same 4 elements.
4. **Compare**: Identify "Dealbreakers" and "Strong Links".

**Output Format:**
You must output a valid JSON object:
{
    "reasoning": "Your step-by-step analysis...",
    "prediction": "A" or "B"
}
"""

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIGS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": "GEMINI_API_KEY_PLACEHOLDER",
        "model": "gemini-2.0-flash"
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key": "MISTRAL_API_KEY_PLACEHOLDER",
        "model": "mistral-small-latest"
    },
    "poe": {
        "base_url": "https://api.poe.com/v1",
        "api_key": "POE_API_KEY_PLACEHOLDER",
        "model": "gpt-4o-mini"
    }
}


# ============================================================================
# MODEL DISCOVERY
# ============================================================================

def find_available_model() -> Tuple[str, str]:
    """
    Find an available embedding model, checking in order of preference.
    
    Returns:
        (model_path_or_name, model_type)
    """
    print("\n" + "="*70)
    print("EMBEDDING MODEL DISCOVERY")
    print("="*70)
    
    # Check A2 fine-tuned model
    a2_path = MODEL_PATHS["a2_finetuned"]
    if a2_path.exists() and (a2_path / "config.json").exists():
        print(f"✅ Found A2 fine-tuned model at: {a2_path}")
        print("   Expected accuracy: ~58.5%")
        return str(a2_path), "a2_finetuned"
    else:
        print(f"❌ A2 fine-tuned model NOT found at: {a2_path}")
    
    # Check A3 multi-task model
    a3_path = MODEL_PATHS["a3_multitask"]
    if a3_path.exists() and (a3_path / "config.json").exists():
        print(f"✅ Found A3 multi-task model at: {a3_path}")
        print("   ⚠️  Warning: This model has ~50.25% accuracy (near random)")
        print("   The hybrid will rely heavily on LLM fallback.")
        return str(a3_path), "a3_multitask"
    else:
        print(f"❌ A3 multi-task model NOT found at: {a3_path}")
    
    # Fall back to pre-trained HuggingFace model
    print(f"✅ Using pre-trained model from HuggingFace: {MODEL_PATHS['pretrained']}")
    print("   Expected accuracy: ~55% (baseline)")
    print("   This will be downloaded automatically if not cached.")
    return MODEL_PATHS["pretrained"], "pretrained"


# ============================================================================
# HYBRID MODEL CLASS
# ============================================================================

class HybridNarrativeSimilarityModel:
    """
    Hybrid model combining fast embeddings with LLM fallback.
    
    Strategy:
    - Use embedding model for quick prediction + confidence score
    - If confidence < threshold, escalate to expensive but accurate LLM
    - Track statistics on path usage for cost analysis
    """
    
    def __init__(
        self,
        embedding_model_path: str = None,
        llm_provider: str = "gemini",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        verbose: bool = True,
        auto_discover: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.llm_provider = llm_provider
        self.verbose = verbose
        self.model_type = None
        
        # Find and load embedding model
        if embedding_model_path:
            model_path = embedding_model_path
            self.model_type = "custom"
        elif auto_discover:
            model_path, self.model_type = find_available_model()
        else:
            raise ValueError("No embedding model specified and auto_discover=False")
        
        if self.verbose:
            print(f"\nLoading embedding model: {model_path}")
        
        try:
            self.embedding_model = SentenceTransformer(model_path)
            if self.verbose:
                print(f"✅ Model loaded successfully!")
                print(f"   Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
        
        # Setup LLM client
        self._setup_llm_client(llm_provider)
        
        # Statistics tracking
        self.stats = {
            "total": 0,
            "fast_path": 0,
            "slow_path": 0,
            "llm_errors": 0,
            "correct": 0,
            "fast_path_correct": 0,
            "slow_path_correct": 0,
            "confidences": [],
            "model_type": self.model_type,
        }
    
    def _setup_llm_client(self, provider: str):
        """Initialize LLM client based on provider."""
        if provider not in API_CONFIGS:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(API_CONFIGS.keys())}")
        
        config = API_CONFIGS[provider]
        self.llm_client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.llm_model = config["model"]
        
        if self.verbose:
            print(f"\nLLM Configuration:")
            print(f"   Provider: {provider}")
            print(f"   Model: {self.llm_model}")
            print(f"   Confidence threshold: {self.confidence_threshold}")
    
    def _get_embedding_prediction(
        self, 
        anchor: str, 
        text_a: str, 
        text_b: str
    ) -> Tuple[bool, float, float, float]:
        """
        Get prediction from embedding model.
        
        Returns:
            (prediction_is_a, confidence, sim_a, sim_b)
        """
        embeddings = self.embedding_model.encode([anchor, text_a, text_b])
        
        sim_a = float(cos_sim(embeddings[0], embeddings[1]).item())
        sim_b = float(cos_sim(embeddings[0], embeddings[2]).item())
        
        prediction_is_a = sim_a > sim_b
        confidence = abs(sim_a - sim_b)
        
        return prediction_is_a, confidence, sim_a, sim_b
    
    def _get_llm_prediction(
        self, 
        anchor: str, 
        text_a: str, 
        text_b: str
    ) -> Tuple[Optional[bool], str]:
        """
        Get prediction from LLM with CoT reasoning.
        
        Returns:
            (prediction_is_a, reasoning)
        """
        user_content = f"Anchor Story:\n{anchor}\n\nStory A:\n{text_a}\n\nStory B:\n{text_b}"
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": COT_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            pred_label = result.get('prediction', '').strip().upper()
            is_a = (pred_label == 'A')
            
            return is_a, result.get('reasoning', 'No reasoning provided')
            
        except Exception as e:
            self.stats["llm_errors"] += 1
            return None, f"Error: {str(e)}"
    
    def predict(
        self, 
        anchor: str, 
        text_a: str, 
        text_b: str,
        ground_truth: bool = None
    ) -> Dict:
        """
        Make prediction using hybrid approach.
        """
        self.stats["total"] += 1
        
        # Stage 1: Fast embedding prediction
        emb_pred, confidence, sim_a, sim_b = self._get_embedding_prediction(
            anchor, text_a, text_b
        )
        
        # Track confidence for distribution analysis
        self.stats["confidences"].append(confidence)
        
        result = {
            "embedding_prediction": emb_pred,
            "confidence": confidence,
            "sim_a": sim_a,
            "sim_b": sim_b,
            "path": None,
            "final_prediction": None,
            "llm_reasoning": None
        }
        
        # Decision: Fast path or slow path?
        if confidence >= self.confidence_threshold:
            # FAST PATH: High confidence, use embedding prediction
            result["path"] = "fast"
            result["final_prediction"] = emb_pred
            self.stats["fast_path"] += 1
            
            if ground_truth is not None and emb_pred == ground_truth:
                self.stats["correct"] += 1
                self.stats["fast_path_correct"] += 1
        else:
            # SLOW PATH: Low confidence, escalate to LLM
            result["path"] = "slow"
            llm_pred, reasoning = self._get_llm_prediction(anchor, text_a, text_b)
            result["llm_reasoning"] = reasoning
            self.stats["slow_path"] += 1
            
            if llm_pred is not None:
                result["final_prediction"] = llm_pred
                if ground_truth is not None and llm_pred == ground_truth:
                    self.stats["correct"] += 1
                    self.stats["slow_path_correct"] += 1
            else:
                # LLM failed, fall back to embedding prediction
                result["final_prediction"] = emb_pred
                result["path"] = "fast_fallback"
                if ground_truth is not None and emb_pred == ground_truth:
                    self.stats["correct"] += 1
        
        return result
    
    def get_stats(self) -> Dict:
        """Get performance statistics with derived metrics."""
        stats = self.stats.copy()
        
        # Remove raw confidences list for cleaner output
        confidences = stats.pop("confidences", [])
        
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
            stats["fast_path_ratio"] = stats["fast_path"] / stats["total"]
            stats["slow_path_ratio"] = stats["slow_path"] / stats["total"]
        
        if stats["fast_path"] > 0:
            stats["fast_path_accuracy"] = stats["fast_path_correct"] / stats["fast_path"]
        else:
            stats["fast_path_accuracy"] = 0.0
            
        if stats["slow_path"] > 0:
            stats["slow_path_accuracy"] = stats["slow_path_correct"] / stats["slow_path"]
        else:
            stats["slow_path_accuracy"] = 0.0
        
        # Confidence statistics
        if confidences:
            import statistics
            stats["confidence_stats"] = {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0,
                "min": min(confidences),
                "max": max(confidences)
            }
        
        # Cost estimation
        stats["estimated_llm_calls"] = stats["slow_path"]
        stats["estimated_cost"] = stats["slow_path"] * 0.001
        stats["cost_vs_full_llm"] = f"{(1 - stats.get('slow_path_ratio', 1)) * 100:.1f}% savings"
        
        return stats
    
    def reset_stats(self):
        """Reset statistics for a new evaluation run."""
        self.stats = {
            "total": 0,
            "fast_path": 0,
            "slow_path": 0,
            "llm_errors": 0,
            "correct": 0,
            "fast_path_correct": 0,
            "slow_path_correct": 0,
            "confidences": [],
            "model_type": self.model_type,
        }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_dev_data(path: str) -> List[Dict]:
    """Load development data from JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_hybrid(
    model: HybridNarrativeSimilarityModel,
    dev_path: str,
    limit: int = None,
    save_results: bool = True
) -> Dict:
    """
    Evaluate hybrid model on development set.
    """
    print(f"\n{'='*70}")
    print("HYBRID MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Dev data: {dev_path}")
    print(f"Embedding model type: {model.model_type}")
    print(f"Confidence threshold: {model.confidence_threshold}")
    print(f"LLM provider: {model.llm_provider}")
    
    # Load data
    dev_data = load_dev_data(dev_path)
    if limit:
        dev_data = dev_data[:limit]
    
    print(f"Samples to evaluate: {len(dev_data)}")
    print(f"{'='*70}\n")
    
    # Run evaluation
    results = []
    for i, item in enumerate(tqdm(dev_data, desc="Evaluating")):
        result = model.predict(
            anchor=item['anchor_text'],
            text_a=item['text_a'],
            text_b=item['text_b'],
            ground_truth=item['text_a_is_closer']
        )
        result["index"] = i
        result["ground_truth"] = item['text_a_is_closer']
        result["correct"] = result["final_prediction"] == item['text_a_is_closer']
        results.append(result)
    
    # Get final stats
    stats = model.get_stats()
    stats["threshold"] = model.confidence_threshold
    stats["llm_provider"] = model.llm_provider
    stats["timestamp"] = datetime.now().isoformat()
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Embedding model:      {stats.get('model_type', 'unknown')}")
    print(f"Total samples:        {stats['total']}")
    print(f"Overall accuracy:     {stats.get('accuracy', 0):.4f} ({stats['correct']}/{stats['total']})")
    print()
    print("PATH ANALYSIS:")
    print(f"  Fast path (embedding):")
    print(f"    - Count:    {stats['fast_path']} ({stats.get('fast_path_ratio', 0)*100:.1f}%)")
    print(f"    - Accuracy: {stats.get('fast_path_accuracy', 0):.4f}")
    print()
    print(f"  Slow path (LLM):")
    print(f"    - Count:    {stats['slow_path']} ({stats.get('slow_path_ratio', 0)*100:.1f}%)")
    print(f"    - Accuracy: {stats.get('slow_path_accuracy', 0):.4f}")
    print(f"    - Errors:   {stats.get('llm_errors', 0)}")
    print()
    print("COST ANALYSIS:")
    print(f"  LLM calls:          {stats['estimated_llm_calls']}")
    print(f"  Estimated cost:     ${stats['estimated_cost']:.4f}")
    print(f"  vs Full LLM:        {stats['cost_vs_full_llm']}")
    print()
    if "confidence_stats" in stats:
        cs = stats["confidence_stats"]
        print("CONFIDENCE DISTRIBUTION:")
        print(f"  Mean:   {cs['mean']:.4f}")
        print(f"  Median: {cs['median']:.4f}")
        print(f"  Stdev:  {cs['stdev']:.4f}")
        print(f"  Range:  [{cs['min']:.4f}, {cs['max']:.4f}]")
    print(f"{'='*70}\n")
    
    # Save results
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Detailed results
        results_file = OUTPUT_DIR / f"hybrid_results_t{model.confidence_threshold}.jsonl"
        with open(results_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        # Summary
        summary_file = OUTPUT_DIR / f"hybrid_summary_t{model.confidence_threshold}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Results saved to: {OUTPUT_DIR}")
    
    return stats


def run_threshold_analysis(
    llm_provider: str = "gemini",
    thresholds: List[float] = None,
    limit: int = None
) -> List[Dict]:
    """
    Run evaluation across multiple thresholds to find optimal.
    """
    if thresholds is None:
        thresholds = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    
    print(f"\n{'='*70}")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Thresholds to test: {thresholds}")
    print(f"{'='*70}\n")
    
    results = []
    
    for thresh in thresholds:
        print(f"\n--- Testing threshold: {thresh} ---")
        
        model = HybridNarrativeSimilarityModel(
            llm_provider=llm_provider,
            confidence_threshold=thresh,
            verbose=False,
            auto_discover=True
        )
        
        stats = evaluate_hybrid(
            model,
            str(DEV_DATA_PATH),
            limit=limit,
            save_results=True
        )
        stats["threshold"] = thresh
        results.append(stats)
    
    # Summary table
    print(f"\n{'='*70}")
    print("THRESHOLD ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Fast %':<12} {'Slow %':<12} {'Cost Savings':<15}")
    print("-" * 63)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r.get('accuracy', 0):<12.4f} "
              f"{r.get('fast_path_ratio', 0)*100:<12.1f} "
              f"{r.get('slow_path_ratio', 0)*100:<12.1f} "
              f"{r.get('cost_vs_full_llm', 'N/A'):<15}")
    
    print("-" * 63)
    
    # Find best threshold
    best = max(results, key=lambda x: x.get('accuracy', 0))
    print(f"\nBest threshold: {best['threshold']} (Accuracy: {best.get('accuracy', 0):.4f})")
    
    # Save analysis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_file = OUTPUT_DIR / "threshold_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Model for Narrative Similarity (Assignment 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 samples
  python musaab_hybrid_model.py --limit 10
  
  # Full evaluation
  python musaab_hybrid_model.py
  
  # Use different LLM provider
  python musaab_hybrid_model.py --provider mistral
  
  # Run threshold analysis
  python musaab_hybrid_model.py --analyze-thresholds
  
  # Force use of A3 model
  python musaab_hybrid_model.py --use-a3-model
  
  # Force use of pretrained model (no local fine-tuning)
  python musaab_hybrid_model.py --use-pretrained
        """
    )
    
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "poe", "mistral"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Confidence threshold for LLM escalation (default: 0.10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for testing")
    parser.add_argument("--analyze-thresholds", action="store_true",
                        help="Run threshold sensitivity analysis")
    parser.add_argument("--use-a3-model", action="store_true",
                        help="Force use of A3 multi-task model")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Force use of pretrained HuggingFace model")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Custom path to embedding model")
    
    args = parser.parse_args()
    
    # Determine embedding model path
    if args.embedding_model:
        model_path = args.embedding_model
    elif args.use_a3_model:
        model_path = str(MODEL_PATHS["a3_multitask"])
        print("Using A3 multi-task model (as requested)")
    elif args.use_pretrained:
        model_path = MODEL_PATHS["pretrained"]
        print("Using pretrained HuggingFace model (as requested)")
    else:
        model_path = None  # Will auto-discover
    
    if args.analyze_thresholds:
        # Run threshold analysis
        run_threshold_analysis(
            llm_provider=args.provider,
            limit=args.limit
        )
    else:
        # Single evaluation
        model = HybridNarrativeSimilarityModel(
            embedding_model_path=model_path,
            llm_provider=args.provider,
            confidence_threshold=args.threshold,
            auto_discover=(model_path is None)
        )
        
        evaluate_hybrid(
            model,
            str(DEV_DATA_PATH),
            limit=args.limit
        )


if __name__ == "__main__":
    main()
# Assignment 2 Report: LLM Reasoner Baselines
**Author:** Muhammad Musaab ul Haq
**Date:** November 28, 2025

## 1. Introduction
This report summarizes the performance of Large Language Model (LLM) baselines for the SemEval 2026 Task 4 (Narrative Similarity) dataset. The objective was to use Chain-of-Thought (CoT) reasoning to identify which of two candidate stories shares the most **structural similarity** with an anchor story.

## 2. Methodology
- **Approach:** Zero-shot Chain-of-Thought (CoT) prompting.
- **Prompt Strategy:** "Structural Similarity" focus. The model was instructed to analyze stories based on:
    1.  Concrete Setting & Time
    2.  Core Conflict Mechanism
    3.  Protagonist Archetype
    4.  Tone/Genre
- **Script:** `src/model_1_llm_reasoner.py` (Refactored to support multiple providers).

## 3. Models Evaluated
Two models were evaluated using the same prompt structure:
1.  **GPT-4o-mini** (via Poe API)
2.  **Mistral Small** (via Mistral API)

## 4. Results

| Model | Samples Processed | Correct Predictions | Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-4o-mini** | 188 | 135 | **71.81%** | 12 samples failed due to API rate limits (Poe). |
| **Mistral Small** | 200 | 138 | **69.00%** | Processed full validation set. |

### 4.1 GPT-4o-mini Performance
- **Accuracy:** 71.81%
- **Observations:** This model achieved the highest accuracy among the tested baselines. However, the run was incomplete (188/200) due to API quota limitations on the Poe platform. The accuracy is calculated based on the successfully processed samples.

### 4.2 Mistral Small Performance
- **Accuracy:** 69.00%
- **Observations:** Mistral Small performed slightly worse than GPT-4o-mini but successfully processed the entire dataset without rate limit interruptions. An accuracy of 69% is a strong baseline, significantly better than random chance (50%).

## 5. Conclusion
Both models demonstrated strong reasoning capabilities using the structural similarity prompt. 
- **GPT-4o-mini** appears to be the superior reasoner for this specific task, achieving ~72% accuracy.
- **Mistral Small** is a viable and cost-effective alternative, offering competitive performance (~69%) with better availability/throughput in this experiment.

Future work could involve:
- Completing the GPT-4o-mini run.
- Experimenting with larger models (e.g., GPT-4o, Claude 3.5 Sonnet).
- Fine-tuning the prompt to address specific failure cases.

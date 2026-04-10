# Agent-Benchmarking
## Experiment Setup & Evaluation Guide
### Overview
This project benchmarks different QA configurations using:
* Hugging Face LLMs
* Optional retrieval augmentation
* Multiple prompt strategies
* LLM-as-a-Judge evaluation
The pipeline includes:
* Answer generation
* Metric evaluation (EM, F1, Faithfulness)
* LLM-based qualitative evaluation
### Setup Instructions
#### 1. Install all the dependencies
* for pipeline:
pip install torch transformers datasets evaluate pandas scikit-learn
* for retrieval:
pip install sentence-transformers faiss-cpu
#### 2. Running Experiments
* Clear Previous Results 
open("results.jsonl", "w").close()
* Run Generation
* py model.py
##### This will
* Load dataset
* Generate answers
* Save results in results.jsonl
* Run LLM-as-a-Judge
* py llm_as_judge.py
##### Outputs:
* Correctness
* Completeness
* Reasoning
#### Experiment Configurations
Modify these parameters inside model_1.py:
1. Model Selection
* model_name = "Qwen/Qwen2.5-0.5B"
2. Prompt Type
* PROMPT_TYPE = "basic"       # or "structured"
* basic → simple QA prompt
* structured → reasoning-based prompt
3. Retrieval Toggle
* use_retrieve = True   # or False
* False → baseline
* True → retrieval-augmented QA
4. Context Length
* context = tokenizer.decode(
    tokenizer(context)["input_ids"][:400]
)
* 200 → faster
* 400 → balanced
* 600 → more context, slower
5. Generation Settings
MAX_NEW_TOKENS = 50
do_sample = False
#### Save results separately:
* results_baseline.jsonl
* results_retrieval.jsonl

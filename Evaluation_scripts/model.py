import time 
import pandas as pd
import torch 
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from evaluate import load
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# model_name="Qwen/Qwen2.5-0.5B"
model_name="Qwen/Qwen3.5-0.8B"
device="cuda" if torch.cuda.is_available() else "cpu"


print(f"Loading model: {model_name} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    torch_dtype="auto"
).to(device)
model.eval()

BATCH_SIZE=20
TOTAL_SAMPLES=1500
MAX_NEW_TOKENS = 30

dataset=load_dataset('squad',split="validation")

samples=dataset.select(range(TOTAL_SAMPLES))

def save_results(batch_preds, batch_refs):
    with open("results_1.jsonl", "a") as f:
        for p, r in zip(batch_preds, batch_refs):
            f.write(json.dumps({"pred": p, "ref": r}) + "\n")

def build_retrieval(dataset_string,top_k):
    loader = HuggingFaceDatasetLoader(dataset_string, "context")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data[:1500])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever

def retrieve_context(retriever,question):
    # Placeholder for a retrieval mechanism. 
    # For simplicity, we return the first 'top_k' contexts from the dataset.
    retrieved_docs = retriever.invoke(question)
    contexts = "\n".join([doc.page_content for doc in retrieved_docs])
    
    return contexts

def faithfulness_score(pred, context):
    return 1 if any(word in context for word in pred.split()) else 0
            
print("starting benchmark...")
total_tokens=0
use_retrieve=False  # Set to True to enable retrieval-augmented generation
faith=0
PROMPT_TYPE="basic"  # Options: "basic", "structered"

if(use_retrieve):
    print("Building retrieval system...")
    retriever = build_retrieval("squad", top_k=2)

start_time=time.time()
for start in range(0, TOTAL_SAMPLES, BATCH_SIZE):
  batch = samples.select(range(start, min(start + BATCH_SIZE, TOTAL_SAMPLES)))

  predictions = []
  references = []
  
  for entry in batch:
    question=entry["question"]
    ground_truth=entry["answers"]["text"][0] if entry["answers"]["text"] else ""
    context=entry["context"]
    if(use_retrieve):
        retrieved_context = retrieve_context(retriever, question)
        context = context[:200] + "\n" + retrieved_context[:200]

    # context
    context = tokenizer.decode(
            tokenizer(context)["input_ids"][:400]
        )
    if(PROMPT_TYPE=="basic"):
       prompt=f"Context: {context}\nQuestion: {question}\nAnswer:"
    else:
       prompt=f"Read the following context and proper reasoning , answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    
    inputs=tokenizer(prompt,return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)
    
    generated_text=tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
    
    num_new_tokens=output_tokens.shape[1]-inputs.input_ids.shape[1]
    total_tokens+=num_new_tokens
    
    faith+=faithfulness_score(generated_text, context)
    predictions.append({
        "id": entry["id"],
        "prediction_text": generated_text.strip()
    })

    references.append({
        "id": entry["id"],
        "answers": entry["answers"]
    })
    
  save_results(predictions, references)

  print(f"Processed batch {start} → {start + BATCH_SIZE}")


end_time=time.time()
print(f"Benchmark completed. Total tokens generated: {total_tokens}")
print(f"Total time taken or Latency: {end_time-start_time} seconds")

metric = load("squad")
preds = []
refs = []

with open("results_1.jsonl") as f:
    for line in f:
        data = json.loads(line)
        preds.append(data["pred"])
        refs.append(data["ref"])

results = metric.compute(predictions=preds, references=refs)
faith_score = faith / TOTAL_SAMPLES
cost = total_tokens * 0.000002  # Assuming $0.002 per 1K tokens for a 0.5B parameter model
results["faithfulness"] = faith_score
results["cost"] = cost
print(results)

summary = {
    "model": model_name,
    "retrieval": use_retrieve,
    "prompt": PROMPT_TYPE,
    "EM": results["exact_match"],
    "F1": results["f1"],
    "latency": end_time - start_time,
    "tokens": total_tokens
}

df = pd.DataFrame([summary])

df.to_csv("experiment_results.csv", mode="a", header=not pd.io.common.file_exists("experiment_results.csv"), index=False)
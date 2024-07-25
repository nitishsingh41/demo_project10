import os
import sys
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from ragas import evaluate
from datasets import load_dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)

# Initialize the language model
llm = ChatGroq(
    temperature=0,
    model="llama-3.1-70b-versatile",
    max_tokens=4000,
    api_key="",
    timeout=2400,
    max_retries=1000
)

# Load the embedding model
embed_model_id = "Alibaba-NLP/gte-large-en-v1.5"
device = 'cuda'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device, 'trust_remote_code': True},
    encode_kwargs={'device': device},
)

# Evaluate the model with different metrics
result1 = evaluate(
    amnesty_qa['eval'], metrics=[answer_relevancy], llm=llm, embeddings=embed_model
)

result2 = evaluate(
    amnesty_qa['eval'], metrics=[context_recall], llm=llm, embeddings=embed_model
)

result3 = evaluate(
    amnesty_qa['eval'], metrics=[answer_similarity], llm=llm, embeddings=embed_model
)

# Combine results into a final dictionary
final_results = {
    "answer_relevancy": result1.get('answer_relevancy', None),
    "context_recall": result2.get('context_recall', None),
    "answer_similarity": result3.get('answer_similarity', None)
}

# Print the final results
print(final_results)

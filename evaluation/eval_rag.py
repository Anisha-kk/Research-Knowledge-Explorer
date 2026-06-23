from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from logger.logger import RKE_Logger

error_logger = RKE_Logger.logger("errorLogger")

judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(
        model="nomic-embed-text"
    )
)

#To assign llama3.2 as the judge llm of ragas
judge_llm = LangchainLLMWrapper(
    ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        num_ctx=4096,   
    )
)

def evaluate_rag(retriever, qa_chain, questions, mode="global"):
    rows = []

    for q in questions:
        question = q["question"]

        results = retriever.retrieve(question, mode=mode)
        docs = [r["doc"] for r in results]

        contexts = [
            doc["text"][:300] if isinstance(doc, dict) else str(doc[:300]) #Sending only 1000 tokens per chunk
            for doc in docs[:3]
        ]

        answer = qa_chain.answer(question, docs)

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q["ground_truth"]
        })
        

    dataset = Dataset.from_list(rows)
    

    run_config = RunConfig(
        timeout=60,      
        max_workers=1      # IMPORTANT for Ollama stability
    )

    try:
        # Generation result evaluation
        faithfulness_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config
        )
        answer_relevancy_result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config
        )
        context_precision_result = evaluate(
            dataset=dataset,
            metrics=[context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config
        )
        context_recall_result = evaluate(
            dataset=dataset,
            metrics=[context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config
        )
        results = {
            "faithfulness": faithfulness_result["faithfulness"],
            "answer_relevancy": answer_relevancy_result["answer_relevancy"],
            "context_precision": context_precision_result["context_precision"],
            "context_recall": context_recall_result["context_recall"],
        }
        return results or None
    
    except Exception as e:
        error_logger.error(f"Evaluation metric computation error: {e}")
        

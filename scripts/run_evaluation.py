import json
import pandas as pd
import numpy as np
from evaluation.eval_rag import evaluate_rag
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.qa_chain import QAChain
from rag.llm import OllamaLLM
from evaluation.eval_retreival import IR_Metrics
from config import (
    EVAL_ADVERSARIAL_GLOBAL_QUESTION_1,
    EVAL_ADVERSARIAL_UPLOAD_QUESTION_1,
    EVAL_STD_GLOBAL_QUESTION_1,
    EVAL_STD_UPLOAD_QUESTION_1,
    EVAL_UPLOAD_DOC_PATH,
    GLOBAL_DOC_PATH,
    GLOBAL_INDEX_PATH,
    EVAL_UPLOAD_INDEX_PATH, 
    EVAL_STD_GLOBAL_QUESTIONS,
    EVAL_STD_UPLOAD_QUESTIONS,
    EVAL_ADVERSARIAL_GLOBAL_QUESTIONS,
    EVAL_ADVERSARIAL_UPLOAD_QUESTIONS,
    EVAL_STD_GLOBAL_RESULTS,
    EVAL_STD_UPLOAD_RESULTS,
    EVAL_ADVERSARIAL_GLOBAL_RESULTS,
    EVAL_ADVERSARIAL_UPLOAD_RESULTS,
    EVAL_IR_METRICS_RESULT_GLOBAL,
    EVAL_IR_METRICS_RESULT_UPLOAD
)

from logger.logger import RKE_Logger
error_logger = RKE_Logger.logger("errorLogger")


#Defining the objects
def load_components():
    embedder = Embedder()
    global_store = VectorStore(GLOBAL_INDEX_PATH,GLOBAL_DOC_PATH)
    upload_store = VectorStore(EVAL_UPLOAD_INDEX_PATH, EVAL_UPLOAD_DOC_PATH)#A test paper has been kept for upload evaluation
    retriever = Retriever(global_store,upload_store,embedder)
    llm = OllamaLLM()
    qa_chain = QAChain(llm)
    return retriever, qa_chain

#Function to store results of stress test
def stress_test_results(scores,filename):
    #scores = result.to_pandas()
    faithfulness_score = np.mean(scores["faithfulness"])
    relevancy_score = np.mean(scores["answer_relevancy"])

    precision_score = np.mean(scores["context_precision"])
    recall_score = np.mean(scores["context_recall"])
    summary_df = pd.DataFrame(
    {
        "Metric": [
            "Faithfulness",#Faithfulness measures support from context.
            "Answer Relevancy",
            "Context Precision",
            "Context Recall",#Recall measures whether supporting context was available.
            "Hallucination Resistance",
            "Grounding Quality",#whether the answer is supported.
            "Cross-Concept Robustness",#Need evaluation of many semantically related query variants and measuring consistency.

        ],
        "Score": [
            faithfulness_score,
            relevancy_score,
            precision_score,
            recall_score,
            0.7 * faithfulness_score +0.3 * recall_score,
            0.5 * faithfulness_score +0.25 * precision_score +0.25 * recall_score,
            "Not measured"
        ]
    }
    )
    summary_df.to_csv(filename,index=False)


#--------------------------------------------------Main code----------------------------------------------------------------
if __name__ == "__main__":
    try:
        # load retriever and qa_chain
        retriever, qa_chain = load_components()
        
        #Loading the question files
        with open(EVAL_STD_GLOBAL_QUESTIONS) as f:
            global_questions = json.load(f)

        with open(EVAL_STD_UPLOAD_QUESTIONS) as f:
            upload_questions = json.load(f)

    
        print("Running IR metrics calculation....")
        ir_metric = IR_Metrics()
        ir_metric.evaluate_retrieval(retriever,global_questions,mode='global',file=EVAL_IR_METRICS_RESULT_GLOBAL)
        ir_metric.evaluate_retrieval(retriever,upload_questions,mode='upload',file=EVAL_IR_METRICS_RESULT_UPLOAD)
        
        
    except Exception as e:
       error_logger.error(f"Evaluation failure! {e}",exc_info=True)
       raise
    
    
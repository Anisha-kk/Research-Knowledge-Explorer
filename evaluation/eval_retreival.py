import numpy as np
import pandas as pd

class IR_Metrics:
    #Recall@k
    def recall_at_k(self,retrieved_ids, relevant_ids, k):
        retrieved_k = retrieved_ids[:k]
        hits5 = len(
            set(retrieved_k)
            &
            set(relevant_ids)
        )
        return hits5 / len(relevant_ids)
    
    #Precision@k
    def precision_at_k(self,retrieved_ids, relevant_ids, k):
        retrieved_k = retrieved_ids[:k]
        hits5 = len(
            set(retrieved_k)
            &
            set(relevant_ids)
        )
        return hits5 / k
    
    #Hit Rate
    def hit_rate(self,retrieved_ids, relevant_ids, k):
        retrieved_k = retrieved_ids[:k]
        return int(
            len(
                set(retrieved_k)
                &
                set(relevant_ids)
            ) > 0
        )
    
    #MRR
    def reciprocal_rank(self,retrieved_ids,relevant_ids):
        for rank, doc_id in enumerate(retrieved_ids,start=1):
            if doc_id in relevant_ids:
                return 1 / rank
        return 0
    
    #NDCG
    def ndcg(self,retrieved_ids,relevant_ids,k):
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1 / np.log2(i + 2)
        ideal_hits5 = min(len(relevant_ids),k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits5))
        return dcg / idcg if idcg else 0
    
    #Evaluation
    def evaluate_retrieval(self,retriever,questions,mode,file):
        recalls5 = []
        recalls10 = []
        precision5 = []
        mrrs = []
        ndcgs10 = []
        hits5 = []
        for q in questions:
            #For k=5 retreival
            results5 = retriever.retrieve(q["question"],mode=mode,k=5)
            retrieved_ids5 = [r["doc_id"] for r in results5]
            relevant_ids5 = q["relevant_docs"]
            #To remove duplicate entries
            retrieved_ids5 = list(dict.fromkeys(retrieved_ids5))
            relevant_ids5 = list(set(q["relevant_docs"]))

            #For k=10 retreival
            results10 = retriever.retrieve(q["question"],mode=mode,k=10)
            retrieved_ids10 = [r["doc_id"] for r in results10]
            relevant_ids10 = q["relevant_docs"]
            #To remove duplicate entries
            retrieved_ids10 = list(dict.fromkeys(retrieved_ids10))
            relevant_ids10 = list(set(q["relevant_docs"]))

            recalls5.append(self.recall_at_k(retrieved_ids5,relevant_ids5,5))
            recalls10.append(self.recall_at_k(retrieved_ids10,relevant_ids10,10))
            precision5.append(self.precision_at_k(retrieved_ids5,relevant_ids5,5))
            mrrs.append(self.reciprocal_rank(retrieved_ids5,relevant_ids5))
            ndcgs10.append(self.ndcg(retrieved_ids10,relevant_ids10,10))
            hits5.append(self.hit_rate(retrieved_ids5,relevant_ids5,5))

        metric_df = pd.DataFrame(
        {
            "Metric": [
                 "Recall@5",
                 "Recall@10", 
                 "Precision@5",
                 "MRR",
                 "NDCG@10",
                 "Hit Rate@5"
            ],
            "Value": [
                np.mean(recalls5),
                np.mean(recalls10),
                np.mean(precision5),
                np.mean(mrrs),
                np.mean(ndcgs10),
                np.mean(hits5)
            ]
        }
        )
        metric_df.to_csv(file,index=False)
        
    


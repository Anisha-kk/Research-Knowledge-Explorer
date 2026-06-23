# AI/ML Research Knowledge Explorer
## Overview
AI/ML Research Knowledge Explorer is an end-to-end Retrieval-Augmented Generation (RAG) system that enables exploring AI/ML research papers through a chat-based interface. The system automatically ingests research papers from academic sources, indexes them using vector embeddings, and enables conversational question-answering using LLM over both a global research corpus and user-uploaded documents.
## Input/Output
**Input:** User query 
<br>For example:
<br>“Define backpropagation”, “Summarize the paper on attention”
<br>The system supports 2 input modes:
1. Global: User can query the system to which the system gives response from a curated research corpus
2. Upload: User can upload a research paper and can ask questions based on it.
   
**Output:** The response from the system
## Algorithm
1. Creating Global Corpus
   - Create a CSV file of important research papers on AI/ML with columns:
     - Title
     - Year of publication
     - Topic
   - Run the ingestion pipeline:
     1. Search paper in OpenAlex
     2. Download metadata and publication details
     3. Fallback to arXiv if unavailable
     4. Download PDFs (if available)
     5. Store metadata
     6. Extract text from PDFs
     7. Chunk documents
     8. Generate embeddings
     9. Store vectors in FAISS

2. Retrieval Process During Query Time
   - Global Mode:
     1. User enters a query
     2. Query is converted into embeddings
     3. Global FAISS vector store is searched for similar chunks
     4. Top-K relevant chunks are retrieved
     5. Context is built using retrieved chunks
     6. Context is passed to the LLM with prompt
     7. LLM generates and returns the response
   - Upload Mode:
     1. User uploads a PDF file
     2. File is extracted, chunked, embedded, and stored in a separate FAISS vector store
     3. User enters a query
     4. Query is converted into embeddings
     5. FAISS vector store for uploaded document is searched
     6. Top-K relevant chunks are retrieved
     7. Context is built using retrieved chunks
     8. Context is passed to the LLM with prompt
     9. LLM generates and returns the response

## Tech Stack
|Component	| Technology |
|----------|------------|
|Frontend	| Streamlit |
|LLM Runtime	| Ollama(llama3.2) |
|Vector Database	| FAISS |
|Embeddings	| Sentence Transformers(all-MiniLM-L6-v2) |
|Academic Data Sources	| OpenAlex, arXiv |
|Language	| Python |

## Running the system
**Install Dependencies**<br>
pip install -r requirements.txt

**Install Ollama**<br>
https://ollama.com<br>
Pull a model:<br>
ollama pull llama3.2<br>
Verify:<br>
ollama list

**Run ingestion**<br>
python -m scripts.run_ingestion

**Run the application**<br>
streamlit run app.py<br>
Open:<br>
http://localhost:8501

## Evaluation
A custom benchmark dataset was created using manually curated query-document relevance mappings. Two separate datasets, one each for each mode (global/upload) were created with 20 questions and 15 questions respectively.<br>
Retrieval performance was evaluated using:<br>

**Precision@K**<br>
Measures the proportion of relevant documents retrieved in the top-K results.<br>
Precision@K = Number of Relevant Documents in Top-K  / K<br>

**Recall@K**<br>
Measures the percentage of all relevant documents successfully retrieved.<br>
Recall@K= Number of Relevant Documents in Top-K / Total Number of Relevant Documents<br>

**Mean Reciprocal Rank(MRR)**<br>
Evaluates how quickly the first relevant document appears.<br>
MRR=Average of reciprocal ranks (1/rank) over the queries <br>

**Normalized Discounted Cumulative Gain (NDCG)**<br>
Measures ranking quality while accounting for document relevance positions.<br>
0≤NDCG≤1<br>
•  1 = perfect ranking<br> 
•  0 = poor ranking<br>

**Hit Rate(Hit@K)**<br>
Measures how often at least one relevant document appears in the top-K retrieved results.<br>

**Run evaluation script**<br>
python -m scripts.run_evaluation

### Result Interpretation

#### Global Corpus Retrieval Performance

| Metric	| Value	| Interpretation |
|--------|-------|----------------|
|Recall@5	| 0.408	| The top 5 retrieved results contain about 41% of all relevant documents available in the corpus. |
|Recall@10 | 0.498 |	Only 9% improvement from Recall@5 => Most useful results are already appearing in top 5. |
|Precision@5	| 0.28	| Among the top 5 retrieved chunks, only 28% are relevant |
|NDCG@10	| 0.52	| Ranking quality is moderate. Relevant documents are generally near the top, but not always in the optimal order. |
|Hit Rate @5	| 0.95	| 95% of queries retrieved at least one relevant document within the top 5 results. |
|MRR	| 0.775	| Relevant documents are appearing very early in the ranking (rank 1 or 2) |

The FAISS-based semantic retrieval system achieved a Hit Rate@5 of 95% and an MRR of 0.775, indicating that relevant research documents were successfully retrieved for most queries and typically appeared within the top-ranked results. Retrieval effectiveness was further evaluated using Recall@K, Precision@K, and NDCG, demonstrating strong retrieval coverage with opportunities for improved ranking precision through reranking and hybrid retrieval strategies.

#### Upload Retrieval Performance
A test paper was indexed and stored in vector store for evaluating system performance on uploaded file.
| Metric	| Value	| Interpretation |
|--------|-------|----------------|
|Recall@5	| 1.0	| All relevant chunks were successfully retrieved within the top 5 results. |
|Recall@10	| 1.0	| The retriever had already found all relevant content within the first 5 results. |
|Precision@5	| 0.2	| Maybe number of relevant chunks are less.. Returns extra chunks that are not labeled relevant |
|NDCG@10	| 1.0	| The ranking order is perfectly aligned with the ideal ranking. |
|Hit Rate @5	| 1.0	| The uploaded paper was successfully found for every query. |
|MRR	| 1.0	| The most relevant chunk was always the top-ranked result. |

For user-uploaded research papers, the FAISS-based retrieval system achieved perfect retrieval effectiveness, with Recall@5 = 1.0, MRR = 1.0, NDCG@10 = 1.0, and Hit Rate@5 = 1.0, demonstrating that relevant document chunks were consistently retrieved and ranked first. Precision@5 of 0.20 indicates that additional non-relevant chunks were included in the retrieved set, prioritizing retrieval coverage over aggressive filtering.<br>
Due to computational constraints of local CPU-only evaluation, retrieval quality was assessed using standard Information Retrieval metrics (Hit@K, Precision@K, Recall@K, MRR, Hit Rate and NDCG) instead of large-scale LLM-based faithfulness evaluation.

## Examples
Home page:
 <img width="1920" height="943" alt="Screenshot (5027)" src="https://github.com/user-attachments/assets/81250854-e932-4277-8468-b38ce4dbcfee" />

Global corpus query and answer:
 <img width="1920" height="1017" alt="Screenshot (5029)" src="https://github.com/user-attachments/assets/3acdbe03-2053-43a2-8907-b413e48595b4" />

Uploading a paper:
<img width="1920" height="984" alt="Screenshot (5031)" src="https://github.com/user-attachments/assets/0713bcef-91ae-47cf-9e54-b81928f4c63a" />

 
Query and answer based on uploaded file:
<img width="1920" height="1000" alt="Screenshot (5033)" src="https://github.com/user-attachments/assets/68e2f3ca-850c-4ef2-9c89-41f608ae8d31" />


## Key Capabilities

- Academic paper discovery
- Automated metadata ingestion
- Semantic document retrieval
- Conversational research assistant
- Research paper summarization
- Local LLM deployment with Ollama
- User document uploads
- IR-based retrieval evaluation

## Limitations

- Only PDF documents are supported for upload
- Only one document can be uploaded at a time
- The global research corpus currently contains 60 AI/ML papers, limiting coverage of broader research topics
- Retrieval quality depends on the relevance and completeness of the indexed corpus; questions outside the corpus scope may return insufficient results
- FAISS retrieval is based on dense embedding similarity only and does not currently use hybrid retrieval (e.g., BM25 + vector search)
- Retrieved chunks are not reranked using a cross-encoder, which may affect ranking precision
- Evaluation was performed on a relatively small benchmark dataset, which may limit the statistical significance of the reported metrics
- LLM-based RAG evaluation metrics such as faithfulness were not extensively used due to CPU-only hardware constraints and high evaluation latency
- Uploaded documents are re-indexed independently and do not persist across application resets unless stored explicitly
- The system currently supports only English-language research papers
- PDF text extraction quality depends on the source document; scanned PDFs and image-based papers may produce incomplete text extraction
- The QA system relies solely on retrieved context and may return "Not found in provided documents" when relevant information is fragmented across chunks
- No metadata filtering (author, year, venue, research area) is currently implemented during retrieval
- Local inference through Ollama may produce slower response times compared to cloud-hosted LLMs, especially on CPU-only systems

## Future Scope

- Support multiple document uploads
- Expand the global corpus to several hundred or thousand research papers
- Add hybrid retrieval (BM25 + FAISS)
- Introduce reranking models for improved precision
- Support scanned PDFs through OCR
- Add metadata-based filtering and advanced search
- Enable persistent user-specific document collections
- Incorporate larger evaluation datasets and comprehensive RAG metrics when adequate compute resources are available
- Agentic Research Workflows
- Knowledge Graph Integration
- GPU-based Embedding Generation 




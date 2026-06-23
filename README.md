#AI/ML Research Knowledge Explorer
##Overview
AI/ML Research Knowledge Explorer is an end-to-end Retrieval-Augmented Generation (RAG) system that enables summarizing and querying of AI/ML research papers through a chat-based interface. The system automatically ingests research papers from academic sources, indexes them using vector embeddings, and provides conversational question-answering using LLM over both a global research corpus and user-uploaded documents.
##Input/Output
**Input:** User query 
For example: “Define backpropagation”, “Summarize the paper on attention”
The system supports 2 input modes:
•	Global: User can query the system to which the system gives response from a curated research corpus 
•	Upload: User can upload a research paper and can query the system based on it
**Output:** The response from the system
##Algorithm
1.	Creating Global Corpus
a.	Create a CSV file of important research papers on AI/ML with the columns title, year of publication and topic
b.	Run the ingestion pipeline:
i.	Search paper in OpenAlex
ii.	Download metadata and publication details
iii.	Fallback to arXiv if unavailable
iv.	Download PDFs (if available)
v.	Store metadata
vi.	Extract text from PDFs
vii.	Chunk documents
viii.	Generate embeddings
ix.	Store vectors in FAISS
2.	Retreival process during querying time
a.	Global mode:
i.	User enters the query
ii.	Query is embedded
iii.	Global FAISS Vector store is searched for similar chunks
iv.	Top k chunks are retrieved
v.	Context is build using the chunks
vi.	Context is passed with prompt to the LLM. If query contain the term “summarize”, the term “summarize” is also added to the prompt
vii.	LLM returns the response
b.	Upload mode:
i.	User uploads the pdf file 
ii.	The file is extracted, chunked, embedded and stored in a separate vector store
iii.	User enters the query
iv.	Query is embedded
v.	FAISS Vector store for the uploaded file is searched for similar chunks
vi.	Top k chunks are retrieved
vii.	Context is build using the chunks
viii.	Context is passed with prompt to the LLM. If query contain the term “summarize”, the term “summarize” is also added to the prompt
ix.	LLM returns the response

##Tech Stack
Component	Technology
Frontend	Streamlit
LLM Runtime	Ollama(llama3.2)
Vector Database	FAISS
Embeddings	Sentence Transformers(all-MiniLM-L6-v2)
Academic Data Sources	OpenAlex, arXiv
Language	Python

##Running the system
Install Dependencies
pip install -r requirements.txt

Install Ollama:
https://ollama.com
Pull a model:
ollama pull llama3.2
Verify:
ollama list

Run ingestion:
python -m scripts.run_ingestion

Run the application

streamlit run app.py
Open:
http://localhost:8501
##Evaluation
A custom benchmark dataset was created using manually curated query-document relevance mappings. Two separate datasets, one each for each mode (global/upload) were created with 20 questions and 15 questions respectively.
Retrieval performance was evaluated using:
Precision@K
Measures the proportion of relevant documents retrieved in the top-K results.
Precision@K = Number of Relevant Documents in Top-K  / K
Recall@K
Measures the percentage of all relevant documents successfully retrieved.
Recall@K= Number of Relevant Documents in Top-K / Total Number of Relevant Documents
Mean Reciprocal Rank (MRR)
Evaluates how quickly the first relevant document appears.
MRR=Average of reciprocal ranks (1/rank) over the queries 
Normalized Discounted Cumulative Gain (NDCG)
Measures ranking quality while accounting for document relevance positions.
0≤NDCG≤1
•  1 = perfect ranking 
•  0 = poor ranking
Hit Rate(Hit@K)
Measures how often at least one relevant document appears in the top-K retrieved results.
Run evaluation script:
python -m scripts.run_evaluation

###Result Interpretation

####Global Corpus Retrieval Performance

Metric	Value	Interpretation
Recall@5	0.408	The top 5 retrieved results contain about 41% of all relevant documents available in the corpus.
Recall@10	0.498	Only 9% improvement from Recall@5 => Most useful results are already appearing in top 5.
Precision@5	0.28	Among the top 5 retrieved chunks, only 28% are relevant
NDCG@10	0.52	Ranking quality is moderate. Relevant documents are generally near the top, but not always in the optimal order.
Hit Rate @5	0.95	95% of queries retrieved at least one relevant document within the top 5 results.
MRR	0.775	Relevant documents are appearing very early in the ranking (rank 1 or 2)

The FAISS-based semantic retrieval system achieved a Hit Rate@5 of 95% and an MRR of 0.775, indicating that relevant research documents were successfully retrieved for most queries and typically appeared within the top-ranked results. Retrieval effectiveness was further evaluated using Recall@K, Precision@K, and NDCG, demonstrating strong retrieval coverage with opportunities for improved ranking precision through reranking and hybrid retrieval strategies.
####Upload Retrieval Performance
A test paper was indexed and stored in vector store for evaluating system performance on uploaded file.
Metric	Value	Interpretation
Recall@5	1.0	All relevant chunks were successfully retrieved within the top 5 results.
Recall@10	1.0	The retriever had already found all relevant content within the first 5 results.
Precision@5	0.2	Maybe number of relevant chunks are less.. Returns extra chunks that are not labeled relevant
NDCG@10	1.0	The ranking order is perfectly aligned with the ideal ranking.
Hit Rate @5	1.0	The uploaded paper was successfully found for every query.
MRR	1.0	The most relevant chunk was always the top-ranked result.

For user-uploaded research papers, the FAISS-based retrieval system achieved perfect retrieval effectiveness, with Recall@5 = 1.0, MRR = 1.0, NDCG@10 = 1.0, and Hit Rate@5 = 1.0, demonstrating that relevant document chunks were consistently retrieved and ranked first. Precision@5 of 0.20 indicates that additional non-relevant chunks were included in the retrieved set, prioritizing retrieval coverage over aggressive filtering.
Due to computational constraints of local CPU-only evaluation, retrieval quality was assessed using standard Information Retrieval metrics (Hit@K, Precision@K, Recall@K, MRR, Hit Rate and NDCG) instead of large-scale LLM-based faithfulness evaluation.
Examples
Home page:
 

Global corpus query and answer
 
Uploading a paper
 
Query and answer based on uploaded file
 
Key Capabilities
•	Academic paper discovery
•	Automated metadata ingestion
•	Semantic document retrieval
•	Conversational research assistant
•	Research paper summarization
•	Local LLM deployment with Ollama
•	User document uploads
•	IR-based retrieval evaluation
Limitations
•	Only PDF documents are supported for upload. 
•	Only one document can be uploaded at a time. 
•	The global research corpus currently contains 60 AI/ML papers, limiting coverage of broader research topics.
•	Retrieval quality depends on the relevance and completeness of the indexed corpus; questions outside the corpus scope may return insufficient results. 
•	FAISS retrieval is based on dense embedding similarity only and does not currently use hybrid retrieval (e.g., BM25 + vector search). 
•	Retrieved chunks are not reranked using a cross-encoder, which may affect ranking precision. 
•	Evaluation was performed on a relatively small benchmark dataset, which may limit the statistical significance of the reported metrics. 
•	LLM-based RAG evaluation metrics such as faithfulness were not extensively used due to CPU-only hardware constraints and high evaluation latency. 
•	Uploaded documents are re-indexed independently and do not persist across application resets unless stored explicitly. 
•	The system currently supports only English-language research papers. 
•	PDF text extraction quality depends on the source document; scanned PDFs and image-based papers may produce incomplete text extraction. 
•	The QA system relies solely on retrieved context and may return "Not found in provided documents" when relevant information is fragmented across chunks. 
•	No metadata filtering (author, year, venue, research area) is currently implemented during retrieval. 
•	Local inference through Ollama may produce slower response times compared to cloud-hosted LLMs, especially on CPU-only systems.

Future Scope
•	Support multiple document uploads. 
•	Expand the global corpus to several hundred or thousand research papers. 
•	Add hybrid retrieval (BM25 + FAISS). 
•	Introduce reranking models for improved precision. 
•	Support scanned PDFs through OCR. 
•	Add metadata-based filtering and advanced search. 
•	Enable persistent user-specific document collections. 
•	Incorporate larger evaluation datasets and comprehensive RAG metrics when adequate compute resources are available.
•	Agentic Research Workflows
•	Knowledge Graph Integration
•	GPU-based Embedding Generation




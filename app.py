import traceback
import time
import streamlit as st

from utils.file_io import save

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.qa_chain import QAChain
from rag.summarizer import Summarizer
from rag.llm import OllamaLLM

from config import GLOBAL_DOC_PATH,GLOBAL_INDEX_PATH,UPLOAD_DOC_PATH,UPLOAD_INDEX_PATH

from logger.logger import RKE_Logger
info_logger = RKE_Logger.logger("infoLogger")
error_logger = RKE_Logger.logger("errorLogger")

info_logger.info("Starting the application...")
try:
    # -------------------------
    # INIT COMPONENTS
    # -------------------------
    @st.cache_resource
    def load_components():
        embedder = Embedder()

        global_store = VectorStore(
            GLOBAL_INDEX_PATH,
            GLOBAL_DOC_PATH
        )

        upload_store = VectorStore(
            UPLOAD_INDEX_PATH,
            UPLOAD_DOC_PATH
        )

        retriever = Retriever(global_store, upload_store, embedder)

    
        llm = OllamaLLM()

        qa_chain = QAChain(llm)
        summarizer = Summarizer(llm)

        return retriever, qa_chain, summarizer


    retriever, qa_chain, summarizer = load_components()


    # -------------------------
    # SESSION STATE
    # -------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []


    # -------------------------
    # UI
    # -------------------------
    st.title("📚 AI/ML Research Knowledge Explorer")

    uploaded_file = st.file_uploader("Upload paper (optional)", type=["pdf"])

    query = st.chat_input("Ask about AI / ML papers...")


    # --------------
    # HANDLE UPLOAD 
    # --------------
    if uploaded_file is not None:
        with st.spinner("Uploading and indexing paper..."):
            info_logger.info("File uploaded")
            file_id = uploaded_file.name
            # prevent duplicate indexing
            existing = [d["id"] for d in st.session_state.uploaded_docs]

            if file_id not in existing:
                pdf_path = save(uploaded_file)
                info_logger.info(f"Uploaded the file {uploaded_file.name}")
            
                # Applying ingestion logic to store uploaded paper in vector store
                from ingestion.extractor import PDFExtractor
                from ingestion.chunker import Chunker

                extractor = PDFExtractor()
                chunker = Chunker()

                text = extractor.extract(pdf_path)
                chunks = chunker.chunk(text)
                vectors = retriever.embedder.embed(chunks)#Embedding the uploaded pdf

                docs = [{
                    "text": c,
                    "source": "user_upload",
                    "pdf_path": pdf_path,
                    "chunk_id": i,
                    "title":""
                } for i, c in enumerate(chunks)]

                retriever.upload_store.clear() #Clearing previous data
                retriever.upload_store.add(vectors, docs)

                st.session_state.uploaded_docs.append({
                    "id": file_id,
                    "path": pdf_path
                })
            
            info_logger.info("Paper indexed successfully...")
            st.success("Paper indexed successfully")


    # -------------------------
    # DISPLAY CHAT HISTORY
    # -------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


    # -------------------------
    # PROCESS QUERY
    # -------------------------
    if query:

        info_logger.info(f"Query: {query}")
        st.session_state.messages.append({"role": "user", "content": query})

        # Display it immediately
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()

        # Show temporary message
        placeholder.markdown("Searching papers and generating answer...")

        # smarter mode selection
        if len(st.session_state.uploaded_docs) > 0:
            mode = "upload"
        else:
            mode = "global"

        info_logger.info(f"Mode of file: {mode}")
        
        results = retriever.retrieve(query, mode=mode)
        # handle None safely
        if not results:
            info_logger.info("No results from retriever")
            st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry! I do not have answer to this question..."
                })
            st.rerun()
        docs = [r["doc"] for r in results]
        
        #Handling empty output
        if not docs:
            error_logger.info("No answer to the query")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry! I do not have answer to this question..."})
            st.rerun()

    
        #Passing the retrieved data to llm for augmented generation
        if "summarize" in query.lower():
            answer = summarizer.summarize(docs)
        else:
            answer = qa_chain.answer(query, docs)

        info_logger.info(f"Answer:{answer}")

        # Replace Thinking... with actual answer
        placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()
    

except Exception as e:
    error_logger.error(f"Application failed! {traceback.format_exc()}")
    raise
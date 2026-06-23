from logger.logger import RKE_Logger
info_logger = RKE_Logger.logger("infoLogger")
error_logger = RKE_Logger.logger("errorLogger")
import os
from ingestion.pipeline import IngestionPipeline

def ingest_local_pdf(title, pdf_path, year=None):
    """
    Directly ingest a local PDF without OpenAlex/arXiv.
    """
    ingest = IngestionPipeline()
    if not os.path.exists(pdf_path):
        error_logger.error(f"Local PDF not found: {pdf_path}")
        return

    if not ingest.is_valid_pdf(pdf_path):
        error_logger.warning(f"Invalid local PDF: {title}")
        return

    text = ingest.extractor.extract(pdf_path)

    if not text or len(text.strip()) < 500:
        error_logger.warning(f"Too little text in local PDF: {title}")
        return

    chunks = ingest.chunker.chunk(text)

    if not chunks:
        error_logger.warning(f"No chunks: {title}")
        return

    vectors = ingest.embedder.embed(chunks)

    docs = [
        {
            "chunk_id": i,
            "text": c,
            "title": title,
            "pdf_path": pdf_path,
            "source": "local"
        }
        for i, c in enumerate(chunks)
    ]

    ingest.vstore.add(vectors, docs)

    ingest.state.mark_done(title, {
        "pdf": pdf_path,
        "chunks": len(chunks),
        "doi": None,
        "id": None,
        "source": "local"
    })

    info_logger.info(f"Locally ingested: {title}")

if __name__=="__main__":
    ingest_local_pdf(
    title="Attention Is All You Need",
    pdf_path="data/papers/global/attention_is_all_you_need.pdf",
    year=2017
)
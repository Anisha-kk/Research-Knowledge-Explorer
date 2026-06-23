import os
import fitz

from ingestion.csv_loader import CSVLoader
from ingestion.openalex_client import OpenAlexClient
from ingestion.arxiv_fallback import ArxivClient
from ingestion.pdf_downloader import PDFDownloader
from ingestion.extractor import PDFExtractor
from ingestion.chunker import Chunker
from ingestion.state import StateManager

from rag.embedder import Embedder
from rag.vector_store import VectorStore

from logger.logger import RKE_Logger
from config import GLOBAL_DOC_PATH, GLOBAL_INDEX_PATH


info_logger = RKE_Logger.logger("infoLogger")
error_logger = RKE_Logger.logger("errorLogger")


class IngestionPipeline:

    def __init__(self):
        self.loader = CSVLoader("data/papers.csv")
        self.openalex = OpenAlexClient()
        self.arxiv = ArxivClient()
        self.downloader = PDFDownloader()
        self.extractor = PDFExtractor()
        self.chunker = Chunker()
        self.state = StateManager()

        self.embedder = Embedder()
        self.vstore = VectorStore(GLOBAL_INDEX_PATH, GLOBAL_DOC_PATH)

        # micro-batching buffers
        self.batch_vectors = []
        self.batch_docs = []
        self.BATCH_SIZE = 256

    # -------------------------
    # PDF VALIDATION
    # -------------------------
    def is_valid_pdf(self, path):
        if not path or not os.path.exists(path):
            return False

        if os.path.getsize(path) < 50_000:
            return False

        try:
            doc = fitz.open(path)

            if doc.page_count < 1:
                return False

            sample_text = ""
            pages_to_check = min(5, doc.page_count)

            for i in range(pages_to_check):
                sample_text += doc[i].get_text()

            return len(sample_text.strip()) >= 400

        except Exception:
            return False

    # -------------------------
    # FLUSH BATCH TO FAISS
    # -------------------------
    def flush_batch(self):
        if not self.batch_vectors:
            return

        self.vstore.add(self.batch_vectors, self.batch_docs)

        self.batch_vectors = []
        self.batch_docs = []
    #Normalise the text
    def normalize_title(self, title):
        return " ".join(title.lower().strip().split())
    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def run(self):
        print("Starting ingestion...")

        papers = self.loader.load()

        for paper in papers:
            try:
                title = paper["title"]
                year = int(paper["year"])

                title_clean = self.normalize_title(title)

                if self.state.is_done(title_clean):
                    continue

                meta = self.openalex.search(title_clean, year)

                if not meta or not meta.get("pdf_url"):
                    info_logger.info(f"Fallback to arXiv: {title}")
                    meta = self.arxiv.search_arxiv(title_clean, year)

                if not meta:#No meta in OpenAlex and Arxiv
                    error_logger.warning(f"No metadata found: {title}")
                    continue

                pdf_url = meta.get("pdf_url")
                if not pdf_url:
                    error_logger.warning(f"No PDF URL: {title}")
                    continue

                info_logger.info(f"Downloading: {pdf_url}")

                pdf_path = self.downloader.download(pdf_url, title)

                if not pdf_path or not os.path.exists(pdf_path):
                    error_logger.error(f"Download failed: {title}")
                    continue

                if not self.is_valid_pdf(pdf_path):
                    error_logger.warning(f"Invalid PDF: {title}")
                    continue

                text = self.extractor.extract(pdf_path)

                if not text or len(text.strip()) < 3000:
                    error_logger.warning(f"Bad extraction: {title}")
                    continue

                chunks = self.chunker.chunk(text)

                if not chunks or len(chunks) < 5:
                    error_logger.warning(f"Bad chunks: {title}")
                    continue

                # -------------------------
                # FAST BATCH EMBEDDING
                # -------------------------
                vectors = self.embedder.embed(chunks)

                # -------------------------
                # BUILD DOCS + BUFFER
                # -------------------------
                for i, chunk in enumerate(chunks):
                    self.batch_docs.append({
                        "chunk_id": i,
                        "text": chunk,
                        "title": title,
                        "pdf_path": pdf_path,
                        "source": "openalex/arxiv"
                    })

                    self.batch_vectors.append(vectors[i])

                # -------------------------
                # FLUSH IF BATCH FULL
                # -------------------------
                if len(self.batch_vectors) >= self.BATCH_SIZE:
                    self.flush_batch()

                # -------------------------
                # STATE UPDATE
                # -------------------------
                self.state.mark_done(
                    title_clean,
                    {
                        "pdf": pdf_path,
                        "chunks": len(chunks),
                        "doi": meta.get("doi") or None,
                        "id": meta.get("id") or None,
                        "source": meta.get("source") or None
                    }
                )

                info_logger.info(f"Ingested: {title}")

            except Exception as e:
                error_logger.error(f"Failed {paper.get('title','unknown')}: {e}")

        # flush remaining data
        self.flush_batch()
        print("Ingestion complete.")
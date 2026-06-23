import fitz  # PyMuPDF
from logger.logger import RKE_Logger
error_logger = RKE_Logger.logger("errorLogger")

class PDFExtractor:
    def extract(self, path):
        try:
            doc = fitz.open(path)
            text = ""

            for page in doc:
                text += page.get_text()

            return text
        except Exception as e:
            error_logger.error(f"Extractor failure: {e}")
            return None


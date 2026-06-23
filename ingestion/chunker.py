from logger.logger import RKE_Logger
error_logger = RKE_Logger.logger("errorLogger")

class Chunker:
    def chunk(self, text, size=800, overlap=100):
        try:
            chunks = []
            i = 0

            while i < len(text):
                chunks.append(text[i:i+size])
                i += size - overlap

            return chunks
        except Exception as e:
            error_logger.error(f"Chunker failure: {e}")
            return None

import pandas as pd
from logger.logger import RKE_Logger
error_logger = RKE_Logger.logger("errorLogger")

class CSVLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        try:
            df = pd.read_csv(self.path)
            df.columns = [c.lower() for c in df.columns]

            papers = []
            for _, row in df.iterrows():
                papers.append({
                    "title": row["title"],
                    "year": row.get("year", None),
                    "topic": row.get("topic", None)
                })
            return papers
        except Exception as e:
            error_logger.error(f"CSVLoader failure: {e}")
            return None

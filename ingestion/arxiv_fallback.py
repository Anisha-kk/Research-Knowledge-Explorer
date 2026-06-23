import feedparser
from rapidfuzz import fuzz
from config import SIMILARITY_THRESHOLD, build_arxiv_url

from logger.logger import RKE_Logger
error_logger = RKE_Logger.logger("errorLogger")


class ArxivClient:
    def __init__(self):
        pass

    def get_arxiv_year(self,entry): #To get the published year

        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                return entry.published_parsed.tm_year

            if hasattr(entry, "published"):
                from datetime import datetime
                return datetime.strptime(
                    entry.published,
                    "%Y-%m-%dT%H:%M:%SZ"
                ).year

        except Exception:
            return None

        return None    

    def search_arxiv(self, title, year=None):
        try:
            query = title.replace(" ", "+")
            url = build_arxiv_url(query)
            feed = feedparser.parse(url)

            if not feed.entries:
                return None

            best = None #Finding out the best result from outputs returned by arxiv
            best_score = 0

            for entry in feed.entries:

                pub_year = self.get_arxiv_year(entry)
                if year and pub_year:
                    if abs(pub_year - year) > 2:
                        continue

                score = fuzz.token_set_ratio(
                    title.lower(),
                    entry.title.lower()
                )
                
                if score > best_score:
                    best_score = score
                    best = entry
                

            if not best:
                return None

            if best_score < SIMILARITY_THRESHOLD:
                return None

            return {
                "title": best.title,
                "pdf_url": best.link.replace(
                    "/abs/",
                    "/pdf/"
                ) + ".pdf",
                "id": best.id,
                "source": "arxiv"
            }

        except Exception as e:

            error_logger.error(f"Arxiv search failed: {e}")
            return None
        

    
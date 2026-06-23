import requests
from config import OPEN_ALEX, SIMILARITY_THRESHOLD
from rapidfuzz import fuzz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logger.logger import RKE_Logger

error_logger = RKE_Logger.logger("errorLogger")


class OpenAlexClient:

    BASE_URL = OPEN_ALEX

    def __init__(self):

        self.session = requests.Session()

        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504]
        )

        adapter = HTTPAdapter(max_retries=retries)

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update({
            "User-Agent": "ResearchKnowledgeExplorer/1.0"
        })

    # -------------------------
    # NORMALIZE TITLE (IMPORTANT)
    # -------------------------
    def normalize(self, text):
        return " ".join(text.lower().strip().split())

    # -------------------------
    # SEARCH
    # -------------------------
    def search(self, title, year=None):
        try:
            title_norm = self.normalize(title)

            params = {
                "search": title,
                "per-page": 10
            }

            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=30
            )

            if not response.text or not response.text.strip():
                error_logger.error("Empty OpenAlex response")
                return None

            if "html" in response.text.lower():
                error_logger.error(f"HTML response: {response.text[:200]}")
                return None

            try:
                data = response.json()
            except Exception:
                error_logger.error(f"Invalid JSON: {response.text[:200]}")
                return None

            results = data.get("results", [])

            best = None
            best_score = 0

            for item in results:

                result_title = item.get("title", "")
                if not result_title:
                    continue

                result_norm = self.normalize(result_title)

                # -------------------------
                # FLEXIBLE YEAR MATCH (FIX)
                # -------------------------
                pub_year = item.get("publication_year")

                if year and pub_year:
                    if abs(pub_year - year) > 2:  
                        continue

                # -------------------------
                # FUZZY MATCH
                # -------------------------
                score = fuzz.token_set_ratio(title_norm, result_norm)

                if score < SIMILARITY_THRESHOLD:
                    continue

                if score > best_score:
                    best_score = score
                    best = item

            if not best:
                return None

            # -------------------------
            # SAFE OA URL EXTRACTION
            # -------------------------
            pdf_url = None

            open_access = best.get("open_access") or {}
            if isinstance(open_access, dict):
                pdf_url = (
                    open_access.get("oa_url")
                    or open_access.get("pdf_url")
                )

            return {
                "title": best.get("title") or None,
                "pdf_url": pdf_url,
                "doi": best.get("doi") or None,
                "id": best.get("id") or None,
                "source": "openalex"
            }

        except Exception as e:
            error_logger.error(f"OpenAlex failure: {e}")
            return None
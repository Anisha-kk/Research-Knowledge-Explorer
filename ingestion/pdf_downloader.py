import os
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import GLOBAL_PAPERS_PATH
from logger.logger import RKE_Logger
error_logger = RKE_Logger.logger("errorLogger")


class PDFDownloader:

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
            "User-Agent": "Mozilla/5.0 ResearchKnowledgeExplorer"
        })

    def sanitize(self, title):

        invalid = '<>:"/\\|?*'

        for c in invalid:
            title = title.replace(c, "_")

        return title

    def download(self, url, title):

        try:

            filename = self.sanitize(title) + ".pdf"

            path = os.path.join(
                GLOBAL_PAPERS_PATH,
                filename
            )

            response = self.session.get(
                url,
                timeout=60,
                stream=True,
                allow_redirects=True
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()

            if "pdf" not in content_type:
                raise ValueError(
                    f"Not a PDF. Content-Type={content_type}"
                )

            with open(path, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    if chunk:
                        f.write(chunk)

            return path

        except Exception as e:

            error_logger.error(
                f"PDF downloader failure: {e}"
            )

            if os.path.exists(path): #Removes corrupted file
                os.remove(path)

            return None
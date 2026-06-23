import uuid
import os
from config import UPLOAD_PAPERS_PATH

def save(pdf_file, folder=UPLOAD_PAPERS_PATH):
    os.makedirs(folder, exist_ok=True)

    unique_name = pdf_file.name
    file_path = os.path.join(folder, unique_name)

    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    return file_path
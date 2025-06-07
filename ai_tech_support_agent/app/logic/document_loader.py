import os
import logging
from pathlib import Path
from typing import Dict, Union

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
# from unstructured.partition.auto import partition_auto # Will be added if specific extractors are insufficient

# Get a logger for this module
logger = logging.getLogger(__name__)

# Placeholder for functions to be implemented
def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        The extracted text, or an empty string if extraction fails.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            logger.warning(f"PDF file is encrypted and cannot be read: {file_path}")
            # Attempt to decrypt with an empty password, though this rarely works.
            try:
                reader.decrypt("")
            except Exception as e:
                logger.error(f"Failed to decrypt PDF {file_path}: {e}")
                return ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        logger.info(f"Successfully extracted text from PDF: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return ""
    return text

def extract_text_from_docx(file_path: Path) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        The extracted text, or an empty string if extraction fails.
    """
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        logger.info(f"Successfully extracted text from DOCX: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return ""
    return text.strip()

def extract_text_from_pptx(file_path: Path) -> str:
    """
    Extracts text from a PPTX file.

    Args:
        file_path: Path to the PPTX file.

    Returns:
        The extracted text, or an empty string if extraction fails.
    """
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        logger.info(f"Successfully extracted text from PPTX: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PPTX {file_path}: {e}")
        return ""
    return text.strip()

def extract_text_from_txt(file_path: Path) -> str:
    """
    Extracts text from a TXT file.

    Args:
        file_path: Path to the TXT file.

    Returns:
        The extracted text, or an empty string if extraction fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Successfully extracted text from TXT: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading TXT file {file_path}: {e}")
        return ""

def extract_text_from_document(file_path: Path) -> str:
    """
    Determines the file type based on its extension and calls the appropriate
    text extraction function.

    Args:
        file_path: Path to the document.

    Returns:
        The extracted text, or an empty string if the file type is unsupported
        or an error occurs.
    """
    extension = file_path.suffix.lower()
    logger.info(f"Attempting to extract text from: {file_path} (type: {extension})")

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    elif extension == ".pptx":
        return extract_text_from_pptx(file_path)
    elif extension == ".txt":
        return extract_text_from_txt(file_path)
    else:
        # Option to use unstructured.partition_auto for other types or as a general fallback
        # from unstructured.partition.auto import partition_auto
        # try:
        #     elements = partition_auto(filename=str(file_path))
        #     text = "\n".join([el.text for el in elements])
        #     logger.info(f"Successfully extracted text using unstructured: {file_path}")
        #     return text
        # except Exception as e:
        #     logger.error(f"Error extracting text with unstructured for {file_path}: {e}")
        #     logger.warning(f"Unsupported file type: {extension} for file {file_path}")
        #     return ""
        logger.warning(f"Unsupported file type: {extension} for file {file_path}. Skipping.")
        return ""

def load_documents_from_directory(directory_path: Path) -> Dict[Path, str]:
    """
    Scans a directory (and its subdirectories) for supported document types
    and extracts text from them.

    Args:
        directory_path: Path to the directory to scan.

    Returns:
        A dictionary where keys are file paths (Path objects) and
        values are the extracted text strings.
    """
    extracted_texts: Dict[Path, str] = {}
    supported_extensions = [".pdf", ".docx", ".pptx", ".txt"]

    if not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return extracted_texts

    logger.info(f"Scanning directory for documents: {directory_path}")
    for file_path in directory_path.rglob("*"): # rglob for recursive globbing
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            logger.info(f"Found supported file: {file_path}")
            text = extract_text_from_document(file_path)
            if text: # Only add if text extraction was successful
                extracted_texts[file_path] = text
        elif file_path.is_file():
            logger.debug(f"Skipping unsupported file type: {file_path.suffix} for file {file_path}")

    logger.info(f"Finished scanning directory. Extracted text from {len(extracted_texts)} documents.")
    return extracted_texts

if __name__ == '__main__':
    # This section can be used for basic testing of the module later
    # Note: For this __main__ block to work standalone with proper logging,
    # you'd need to add a basicConfig here, but it's generally not done for library modules.
    # The application using this module (like ingest_documents.py) should set up logging.
    logger.info("Document loader module can be tested here if logging is configured by the caller.")
    # Example:
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    # test_doc_path = Path("path_to_your_test_document.pdf")
    # if test_doc_path.exists():
    #     text = extract_text_from_document(test_doc_path)
    #     logger.info(f"Extracted text (first 100 chars): {text[:100]}")
    # else:
    #     logger.warning(f"Test document not found: {test_doc_path}")
    pass

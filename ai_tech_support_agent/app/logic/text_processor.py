import re
import logging
import nltk
from typing import List, Dict, Any

# Attempt to download 'punkt' if not already available.
# This is for local development; in a deployed environment,
# NLTK data should be pre-installed or managed as part of the deployment.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' tokenizer not found. Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        logging.info("'punkt' downloaded successfully.")
    except Exception as e:
        logging.error(f"Could not download 'punkt': {e}. sent_tokenize may not work.")

from nltk.tokenize import sent_tokenize # word_tokenize can be added if needed later

# Transformers import will be within the function that needs it,
# to avoid import error if the library is not immediately available in all environments
# from transformers import AutoTokenizer

# Configure a logger for this module
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning:
    - Replaces multiple whitespace characters (including newlines, tabs) with a single space.
    - Removes or normalizes other irrelevant characters (can be expanded).
    - Strips leading/trailing whitespace.

    Args:
        text: The input string.

    Returns:
        The cleaned string.
    """
    if not text:
        return ""

    logger.debug(f"Original text (first 100 chars): '{text[:100]}'")

    # Replace multiple whitespace characters (including newlines, tabs, etc.) with a single space
    text = re.sub(r'\s+', ' ', text)

    # Add more specific cleaning rules here if needed, e.g.:
    # text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove non-ASCII characters
    # text = text.lower() # Convert to lowercase - consider if this is always desirable

    text = text.strip() # Remove leading/trailing whitespace

    logger.debug(f"Cleaned text (first 100 chars): '{text[:100]}'")
    return text

def chunk_text_by_paragraph(text: str) -> List[str]:
    """
    Splits text into chunks based on paragraph breaks (two or more newlines).
    Each chunk is then individually cleaned.

    Args:
        text: The input string.

    Returns:
        A list of cleaned text chunks, where each chunk represents a paragraph.
    """
    if not text:
        return []

    logger.debug("Chunking text by paragraph.")
    # Paragraphs are often separated by two or more newlines
    # Split by multiple newlines, keeping the newlines initially, then clean.
    # Using a regex that captures one or more newlines as a delimiter.
    # A simpler split would be `re.split(r'\n\s*\n', text)` if we don't need to clean each chunk.

    raw_paragraphs = re.split(r'(\n\s*){2,}', text) # Split by two or more newlines

    cleaned_paragraphs = []
    for para in raw_paragraphs:
        if para.strip(): # Ensure the paragraph is not just whitespace
            cleaned_para = clean_text(para) # Clean each paragraph
            if cleaned_para: # Ensure cleaning doesn't result in an empty string
                cleaned_paragraphs.append(cleaned_para)

    logger.info(f"Split text into {len(cleaned_paragraphs)} paragraph chunks.")
    return cleaned_paragraphs

def chunk_text_by_sentence(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Splits text into chunks of N sentences.
    Assumes text has been cleaned to some extent (e.g., by clean_text),
    especially regarding excessive whitespace that might affect sentence tokenization.

    Args:
        text: The input string (ideally already cleaned).
        sentences_per_chunk: The number of sentences to group into each chunk.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    if sentences_per_chunk <= 0:
        logger.warning("sentences_per_chunk must be positive. Defaulting to 1.")
        sentences_per_chunk = 1

    logger.debug(f"Chunking text by sentence, {sentences_per_chunk} sentences per chunk.")

    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Failed to tokenize sentences (ensure 'punkt' is downloaded): {e}")
        # Fallback: treat the whole text as one chunk if sentence tokenization fails
        return [text] if text.strip() else []

    chunks = []
    current_chunk_sentences = []
    for i, sentence in enumerate(sentences):
        current_chunk_sentences.append(sentence)
        if (i + 1) % sentences_per_chunk == 0 or (i + 1) == len(sentences):
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []

    logger.info(f"Split text into {len(chunks)} sentence chunks.")
    return chunks

def chunk_text_by_token_count(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens_per_chunk: int = 384, # Max for all-MiniLM-L6-v2 is typically 512, but using a smaller default
    overlap_tokens: int = 50
) -> List[str]:
    """
    Splits text into chunks by token count with overlap, using a specified
    transformer model's tokenizer.

    Args:
        text: The input string.
        model_name: The name of the transformer model for tokenization
                      (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        max_tokens_per_chunk: The target maximum number of tokens per chunk.
                               The actual number might be slightly different due
                               to whole word splitting.
        overlap_tokens: The number of tokens to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    if max_tokens_per_chunk <= overlap_tokens:
        logger.error("max_tokens_per_chunk must be greater than overlap_tokens.")
        # Fallback: return the whole text as one chunk, or handle error differently
        return [text] if text.strip() else []

    logger.debug(f"Chunking text by token count: model='{model_name}', max_tokens={max_tokens_per_chunk}, overlap={overlap_tokens}")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers library not found. Please install it to use token-based chunking.")
        # Fallback: treat the whole text as one chunk
        return [text] if text.strip() else []

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}': {e}")
        # Fallback: treat the whole text as one chunk
        return [text] if text.strip() else []

    tokens = tokenizer.encode(text, add_special_tokens=False) # No special tokens like [CLS], [SEP] for chunking content
    if not tokens:
        return []

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode the chunk tokens back to text
        # Ensure clean decoding, handling potential partial tokens at boundaries if necessary,
        # though `encode` followed by `decode` on slices should generally be safe.
        # `decode` parameters like `clean_up_tokenization_spaces` are tokenizer-specific.
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if chunk_text.strip(): # Ensure the chunk is not just whitespace
            chunks.append(chunk_text)

        if end_idx == len(tokens): # Reached the end of the text
            break

        # Move the start_idx for the next chunk, considering overlap
        start_idx += (max_tokens_per_chunk - overlap_tokens)
        # Ensure start_idx doesn't create tiny last chunks due to overlap logic,
        # though the main loop condition `start_idx < len(tokens)` handles termination.

    logger.info(f"Split text into {len(chunks)} token-based chunks.")
    return chunks

def process_and_chunk_text(
    raw_text: str,
    strategy: str = "token_count",
    **kwargs: Any
) -> List[str]:
    """
    Cleans raw text and then chunks it using the specified strategy.

    Args:
        raw_text: The original text string.
        strategy: The chunking strategy to use.
                  Options: "token_count", "sentence", "paragraph".
        **kwargs: Additional keyword arguments to pass to the specific
                  chunking function (e.g., sentences_per_chunk for "sentence",
                  max_tokens_per_chunk for "token_count").

    Returns:
        A list of processed and chunked text strings.
    """
    logger.info(f"Processing text with strategy: {strategy}")

    # Step 1: Clean the text.
    # For paragraph strategy, cleaning is done within the chunking function
    # to preserve paragraph breaks before splitting.
    if strategy != "paragraph":
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            return []
    else:
        # For paragraph strategy, we pass the raw text, as it handles cleaning internally after splitting.
        cleaned_text = raw_text

    # Step 2: Chunk the text based on the chosen strategy.
    if strategy == "token_count":
        return chunk_text_by_token_count(cleaned_text, **kwargs)
    elif strategy == "sentence":
        return chunk_text_by_sentence(cleaned_text, **kwargs)
    elif strategy == "paragraph":
        # chunk_text_by_paragraph handles its own cleaning of raw_text
        return chunk_text_by_paragraph(raw_text, **kwargs)
    else:
        logger.warning(f"Unknown chunking strategy: {strategy}. Returning empty list.")
        return []

if __name__ == '__main__':
    # For standalone testing of this module, basic logging would need to be configured here.
    # However, as a library module, it assumes the calling application sets up logging.
    logger.info("Text processor module can be tested here if logging is configured by the caller.")
    # Example:
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    # sample_text = "This is a test sentence. This is another one.\n\nThis is a new paragraph. It has two sentences."
    # cleaned = clean_text(sample_text)
    # logger.info(f"Cleaned: '{cleaned}'")
    # chunks_sentence = chunk_text_by_sentence(cleaned)
    # logger.info(f"Sentence chunks: {chunks_sentence}")
    # chunks_token = chunk_text_by_token_count(cleaned) # Requires transformers
    # logger.info(f"Token chunks: {chunks_token}")
    # processed_chunks = process_and_chunk_text(sample_text, strategy="sentence", sentences_per_chunk=2)
    # logger.info(f"Processed (sentence strategy): {processed_chunks}")
    pass

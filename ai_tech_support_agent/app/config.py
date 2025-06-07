import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Initialize a logger for messages specifically from this config module
config_logger = logging.getLogger(__name__)

# --- Environment Variables Loading ---
# Determine the project root directory (assuming config.py is in 'app' subdirectory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Construct the path to the .env file
dotenv_path = PROJECT_ROOT / '.env'

if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    config_logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    config_logger.info(f".env file not found at {dotenv_path}. Using default values or environment-set variables.")

# --- Application Logging Configuration ---
# LOG_LEVEL: Controls the verbosity of application logs.
# Recommended values: DEBUG, INFO, WARNING, ERROR, CRITICAL.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Embedding Model Configuration ---
# MODEL_NAME: The name of the Sentence Transformer model to be used for generating embeddings.
# This model will be downloaded from Hugging Face models hub if not available locally.
# Example: "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-mpnet-base-v2"
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# EMBEDDING_DIMENSION: The dimensionality of the embeddings produced by the chosen MODEL_NAME.
# This value MUST match the output dimension of the specified Sentence Transformer model.
# - "sentence-transformers/all-MiniLM-L6-v2": 384 dimensions
# - "sentence-transformers/paraphrase-MiniLM-L3-v2": 384 dimensions
# - "sentence-transformers/all-mpnet-base-v2": 768 dimensions
# Incorrectly setting this will lead to errors in FAISS index operations.
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))

# --- Vector Store Configuration ---
# VECTOR_STORE_INDEX_DIR: The directory where the FAISS index and its associated metadata
# (like the mapping of IDs to text chunks) will be saved and loaded from.
# Default is './data/vector_store' relative to the project root.
VECTOR_STORE_INDEX_DIR_STR = os.getenv("VECTOR_STORE_INDEX_DIR", "./data/vector_store")
VECTOR_STORE_INDEX_DIR = PROJECT_ROOT / VECTOR_STORE_INDEX_DIR_STR

# TOP_K_RESULTS: The default number of top relevant document chunks to retrieve
# during a search operation if not specified in the query.
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# --- FastAPI Application Configuration ---
# API_HOST: The host address for the FastAPI application.
# "0.0.0.0" makes it accessible from other machines on the network.
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# API_PORT: The port number for the FastAPI application.
API_PORT = int(os.getenv("API_PORT", 8000))

# --- LLM Configuration ---
# LLM_MODEL_NAME: The Hugging Face model name for the language model used for question answering.
# Examples: "google/flan-t5-base", "distilgpt2", "gpt2"
# Ensure you have enough resources (RAM/VRAM) for the chosen model.
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-base")

# LLM_MAX_NEW_TOKENS: The maximum number of new tokens the LLM can generate in its response.
# This helps control the length of the output.
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", 250))

# LLM_TEMPERATURE: Controls the randomness of the LLM's output.
# Lower values (e.g., 0.2) make the output more deterministic and focused.
# Higher values (e.g., 0.8) make it more creative and random.
# A value of 0 often means greedy decoding (most likely token).
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))

# LLM_PROMPT_TEMPLATE: The template used to construct the prompt fed to the LLM.
# It should include placeholders for '{context_chunks}' and '{question}'.
# The default template provides a more robust structure for RAG.
DEFAULT_PROMPT_TEMPLATE = """
You are an AI technical support assistant. Your task is to answer the user's question based *only* on the provided context.
Read the context carefully before answering.

Context:
{context_chunks}

Question: {question}

Based *solely* on the context above, provide a helpful and concise answer.
If the context does not contain the information needed to answer the question, state clearly: "The provided information does not directly answer this question."
Do not make assumptions or use any external knowledge.

Helpful Answer:"""
LLM_PROMPT_TEMPLATE = os.getenv("LLM_PROMPT_TEMPLATE", DEFAULT_PROMPT_TEMPLATE)


# --- Directory Setup ---
# Ensure the directory for the vector store index exists.
# This is important for the application to be able to save the index if it's created.
try:
    VECTOR_STORE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config_logger.info(f"Vector store directory is set to: {VECTOR_STORE_INDEX_DIR}")
    # Check for write access, as it's crucial for saving the index.
    if not os.access(str(VECTOR_STORE_INDEX_DIR), os.W_OK):
        config_logger.warning(
            f"Application may not have write access to the vector store directory: {VECTOR_STORE_INDEX_DIR}. "
            "Index saving might fail. Please check directory permissions."
        )
except Exception as e:
    config_logger.error(
        f"Could not create or access the vector store directory {VECTOR_STORE_INDEX_DIR}: {e}",
        exc_info=True
    )
    # Depending on the application's resilience requirements, you might want to raise an error here
    # or allow it to proceed (e.g., if it only needs to read an existing index).

# --- For Testing or Direct Script Runs ---
if __name__ == '__main__':
    # This section is for directly running this config file to check values.
    # Basic logging for this direct check:
    logging.basicConfig(level="DEBUG", format='%(levelname)s: %(message)s')

    print("\n--- Configuration Values ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"Vector Store Index Directory (Absolute): {VECTOR_STORE_INDEX_DIR.resolve()}")
    print(f"  (Exists: {VECTOR_STORE_INDEX_DIR.exists()}, Is Dir: {VECTOR_STORE_INDEX_DIR.is_dir()})")
    print(f"Top K Results: {TOP_K_RESULTS}")
    print(f"API Host: {API_HOST}")
    print(f"API Port: {API_PORT}")
    print(f"\nLLM Model Name: {LLM_MODEL_NAME}")
    print(f"LLM Max New Tokens: {LLM_MAX_NEW_TOKENS}")
    print(f"LLM Temperature: {LLM_TEMPERATURE}")
    print(f"LLM Prompt Template (first 100 chars): {LLM_PROMPT_TEMPLATE[:100].strip()}...")
    print("--- End of Configuration ---")

    # Test write access again explicitly if running this file
    if VECTOR_STORE_INDEX_DIR.exists() and VECTOR_STORE_INDEX_DIR.is_dir():
        if os.access(str(VECTOR_STORE_INDEX_DIR), os.W_OK):
            config_logger.info(f"Confirmed write access to {VECTOR_STORE_INDEX_DIR}.")
        else:
            config_logger.error(f"Write access test failed for {VECTOR_STORE_INDEX_DIR}.")
    else:
        config_logger.warning(f"Vector store directory {VECTOR_STORE_INDEX_DIR} does not exist or is not a directory.")

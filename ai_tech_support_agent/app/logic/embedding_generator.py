import logging
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional, Union # Union for numpy array return from encode

# Get a logger for this module.
# Assumes that the application using this module (e.g., app/main.py or scripts/ingest_documents.py)
# will configure the root logger.
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Manages the loading and caching of the SentenceTransformer model.
    """
    _model: Optional[SentenceTransformer] = None
    _model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Default model
    _device: str = "cpu"

    @classmethod
    def _initialize(cls, model_name_to_load: Optional[str] = None) -> None:
        """
        Initializes the SentenceTransformer model.
        If model_name_to_load is provided, it overrides the class default for this initialization.
        """
        current_model_name = model_name_to_load if model_name_to_load else cls._model_name

        if cls._model is not None and cls._model_name == current_model_name:
            logger.info(f"Model '{current_model_name}' is already initialized on device '{cls._device}'.")
            return

        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SentenceTransformer model '{current_model_name}' on device '{cls._device}'.")
        try:
            cls._model = SentenceTransformer(current_model_name, device=cls._device)
            cls._model_name = current_model_name # Update current model name if successfully loaded
            logger.info(f"Embedding model '{current_model_name}' loaded successfully on '{cls._device}'.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{current_model_name}': {e}", exc_info=True)
            cls._model = None # Ensure model is None if loading failed
            raise RuntimeError(f"Failed to load SentenceTransformer model '{current_model_name}'.") from e

    @classmethod
    def get_model(cls, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Retrieves the initialized SentenceTransformer model.
        If a model_name is provided and it's different from the currently loaded one,
        it re-initializes the model with the new name.

        Args:
            model_name: Optional. The name of the model to load. If None, uses the current class default.

        Returns:
            The initialized SentenceTransformer model.

        Raises:
            RuntimeError: If the model could not be initialized.
        """
        if model_name and cls._model_name != model_name:
            logger.info(f"Requested model '{model_name}' is different from current '{cls._model_name}'. Re-initializing.")
            cls._model = None # Force re-initialization with the new name
            cls._initialize(model_name)
        elif cls._model is None:
            cls._initialize(cls._model_name) # Initialize with default or previously set name

        if cls._model is None: # Check again in case initialization failed
            raise RuntimeError("Embedding model could not be initialized or is not available.")
        return cls._model

    @classmethod
    def set_model_name(cls, model_name: str) -> None:
        """
        Sets the model name to be used for future initializations.
        If the new model name is different from the currently loaded one,
        the current model is unloaded to trigger re-initialization on next `get_model` call.
        """
        if cls._model_name != model_name:
            logger.info(f"Setting default model name to '{model_name}'. Current model '{cls._model_name}' will be replaced on next use if different.")
            if cls._model is not None: # If a model is loaded
                cls._model = None # Unload current model to force re-initialization with new name
            cls._model_name = model_name


def generate_embeddings(
    text_chunks: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using the specified Sentence Transformer model.

    Args:
        text_chunks: A list of text strings to embed.
        model_name: The name of the Sentence Transformer model to use.
                    Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        batch_size: The batch size for encoding. The `encode` method of
                    SentenceTransformer handles batching internally when given a list.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
        Returns an empty list if an error occurs or if no text_chunks are provided.
    """
    if not text_chunks:
        logger.warning("No text chunks provided for embedding generation.")
        return []

    try:
        # Ensure the correct model is loaded, potentially re-initializing if model_name differs.
        model = EmbeddingModel.get_model(model_name=model_name)
    except RuntimeError as e:
        logger.error(f"Failed to get embedding model: {e}", exc_info=True)
        return []

    try:
        logger.info(f"Generating embeddings for {len(text_chunks)} chunks using model '{EmbeddingModel._model_name}' on device '{EmbeddingModel._device}' with batch size {batch_size}...")

        # The encode method can take show_progress_bar=True for console feedback during long tasks
        embeddings = model.encode(
            text_chunks,
            batch_size=batch_size,
            show_progress_bar=False  # Set to True for verbose scripts, False for quieter API/library use
        )

        # Embeddings are numpy arrays; convert them to lists of floats
        embeddings_list = [emb.tolist() for emb in embeddings]
        logger.info(f"Successfully generated {len(embeddings_list)} embeddings.")
        return embeddings_list
    except Exception as e:
        logger.error(f"Error during embedding generation with model '{EmbeddingModel._model_name}': {e}", exc_info=True)
        return []

if __name__ == '__main__':
    # For standalone testing of this module, basic logging would need to be configured here.
    # This allows seeing logs from this module when run directly.
    # In a larger app, logging is typically configured centrally by the main entry point.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Running embedding_generator.py module directly for testing with DEBUG level logging.")

    # Example of setting a different model (optional)
    # EmbeddingModel.set_model_name("sentence-transformers/paraphrase-MiniLM-L3-v2")

    sample_texts = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?"
    ]

    logger.info(f"Test sample texts: {sample_texts}")

    # Generate embeddings using the default model
    embeddings = generate_embeddings(sample_texts)

    if embeddings:
        logger.info(f"Generated {len(embeddings)} embeddings.")
        for i, emb in enumerate(embeddings):
            logger.debug(f"Embedding {i+1} (first 5 dims): {emb[:5]} (Total dims: {len(emb)})")
    else:
        logger.error("Embedding generation failed or returned no results.")

    # Test with an empty list
    logger.info("Testing with empty list of chunks...")
    empty_embeddings = generate_embeddings([])
    if not empty_embeddings:
        logger.info("Correctly returned empty list for empty input.")
    else:
        logger.error(f"Should return empty list for empty input, but got {len(empty_embeddings)} embeddings.")

    # Test trying to load a non-existent model (to check error handling)
    logger.info("Testing with a non-existent model name...")
    try:
        # EmbeddingModel.set_model_name("non-existent-model/foobar") # This sets the default for next get_model if no name passed
        # embeddings_bad_model = generate_embeddings(sample_texts)
        # OR directly:
        embeddings_bad_model = generate_embeddings(sample_texts, model_name="non-existent-model/foobar")
        if not embeddings_bad_model:
            logger.info("Correctly handled non-existent model by returning empty list.")
        else:
            logger.error("Non-existent model test did not behave as expected.")
    except Exception as e: # Catching generic exception for test feedback
        logger.info(f"Caught expected error when trying to load non-existent model: {e}")

    logger.info("Embedding generator module test run complete.")

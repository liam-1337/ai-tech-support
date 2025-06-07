import argparse
import logging
import os # For os.remove
from pathlib import Path
import sys

# Ensure the 'app' directory is in the Python path
# This is often needed when running scripts from a subdirectory if the project isn't installed as a package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.logic.document_loader import load_documents_from_directory
from app.logic.text_processor import process_and_chunk_text
from app.logic.embedding_generator import EmbeddingModel, generate_embeddings
from app.logic.vector_store_manager import VectorStoreManager
from app import config # To use configured MODEL_NAME, EMBEDDING_DIMENSION, etc.

# Configure logging for the script
# This uses the same format as the FastAPI app for consistency.
logging.basicConfig(
    level=config.LOG_LEVEL,  # Use log level from central config
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # handlers=[logging.StreamHandler()] # Ensures logs go to stderr/stdout
)
logger = logging.getLogger(__name__) # Logger for this specific script module

def main():
    """
    Main function to handle document ingestion pipeline.
    Loads documents, processes them into chunks, generates embeddings,
    and adds them to the vector store.
    """
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing documents to ingest."
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="token_count",
        choices=["token_count", "paragraph", "sentence"],
        help="Chunking strategy for processing text. Affects context quality for RAG. Default: token_count."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384, # Default from text_processor
        help="Maximum tokens per chunk if using 'token_count' strategy. Experimentation may be needed for optimal RAG performance. Default: 384."
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=50, # Default from text_processor
        help="Number of overlapping tokens between chunks if using 'token_count' strategy. Helps maintain context. Default: 50."
    )
    parser.add_argument(
        "--sentences-per-chunk",
        type=int,
        default=5, # Default from text_processor
        help="Number of sentences per chunk if using 'sentence' strategy. Alternative to token-based chunking. Default: 5."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generating embeddings. Default: 32."
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="If set, any existing FAISS index and metadata in the configured directory will be deleted and a new one created."
    )
    args = parser.parse_args()

    logger.info("Starting document ingestion process...")
    logger.info(f"Configuration: Input Dir='{args.input_dir}', Strategy='{args.chunk_strategy}', Recreate Index='{args.recreate_index}'")

    # --- 1. Initialize Embedding Model ---
    logger.info(f"Initializing embedding model: {config.MODEL_NAME}")
    try:
        EmbeddingModel.set_model_name(config.MODEL_NAME) # Set the model for the global class instance
        EmbeddingModel.get_model() # Trigger actual model loading
        logger.info(f"Embedding model '{EmbeddingModel._model_name}' loaded on device '{EmbeddingModel._device}'.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        return # Critical failure

    # --- 2. Initialize Vector Store Manager ---
    vector_store_path = config.VECTOR_STORE_INDEX_DIR
    logger.info(f"Initializing vector store at: {vector_store_path} with dimension {config.EMBEDDING_DIMENSION}")

    if args.recreate_index and vector_store_path.exists():
        logger.warning(f"Recreate Index flag is set. Removing existing index files from {vector_store_path}...")
        index_file = vector_store_path / "vector_store.faiss"
        meta_file = vector_store_path / "vector_store_meta.pkl"
        try:
            if index_file.exists():
                os.remove(index_file)
                logger.info(f"Removed existing index file: {index_file}")
            if meta_file.exists():
                os.remove(meta_file)
                logger.info(f"Removed existing metadata file: {meta_file}")
        except OSError as e:
            logger.error(f"Error removing existing index files: {e}", exc_info=True)
            # Decide if to proceed or exit. For now, we'll let VectorStoreManager try to handle it.

    try:
        vector_store = VectorStoreManager(
            index_dir_path=vector_store_path,
            dimension=config.EMBEDDING_DIMENSION # This dimension is used if a new index is created
        )
        if args.recreate_index: # If we just deleted files, ensure manager knows it needs to create.
             vector_store._create_new_index(config.EMBEDDING_DIMENSION)

        logger.info(f"Vector store initialized. Current items: {vector_store.get_ntotal()}, Dimension: {vector_store.get_dimension()}")
        if vector_store.get_dimension() and vector_store.get_dimension() != config.EMBEDDING_DIMENSION:
             logger.warning(f"Dimension Mismatch: Store: {vector_store.get_dimension()}, Config: {config.EMBEDDING_DIMENSION}. This may cause issues.")

    except Exception as e:
        logger.error(f"Failed to initialize vector store manager: {e}", exc_info=True)
        return # Critical failure

    # --- 3. Load Documents ---
    if not args.input_dir.is_dir():
        logger.error(f"Input directory '{args.input_dir}' not found or is not a directory.")
        return

    logger.info(f"Loading documents from: {args.input_dir}")
    try:
        documents_map = load_documents_from_directory(args.input_dir) # Returns Dict[Path, str]
    except Exception as e:
        logger.error(f"Failed to load documents: {e}", exc_info=True)
        return

    if not documents_map:
        logger.info("No documents found in the input directory. Exiting.")
        return
    logger.info(f"Found {len(documents_map)} documents to process.")

    # --- 4. Process, Chunk, Embed, and Add to Store ---
    total_chunks_added = 0
    for doc_path, raw_text in documents_map.items():
        logger.info(f"Processing document: {doc_path.name}")
        if not raw_text.strip():
            logger.info(f"Document {doc_path.name} is empty or contains only whitespace. Skipping.")
            continue

        # Prepare chunking parameters
        chunk_kwargs = {}
        if args.chunk_strategy == "token_count":
            chunk_kwargs['model_name'] = config.MODEL_NAME # Tokenizer should match embedding model
            chunk_kwargs['max_tokens_per_chunk'] = args.max_tokens
            chunk_kwargs['overlap_tokens'] = args.overlap_tokens
        elif args.chunk_strategy == "sentence":
            chunk_kwargs['sentences_per_chunk'] = args.sentences_per_chunk

        logger.debug(f"Chunking '{doc_path.name}' with strategy '{args.chunk_strategy}' and params: {chunk_kwargs}")
        try:
            text_chunks = process_and_chunk_text(
                raw_text=raw_text,
                strategy=args.chunk_strategy,
                **chunk_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to process and chunk document {doc_path.name}: {e}", exc_info=True)
            continue # Skip to next document

        if not text_chunks:
            logger.info(f"No text chunks produced for document {doc_path.name}. Skipping.")
            continue
        logger.info(f"Produced {len(text_chunks)} chunks for document {doc_path.name}.")

        # Generate embeddings for the chunks
        logger.debug(f"Generating embeddings for {len(text_chunks)} chunks from '{doc_path.name}' (batch size: {args.batch_size}).")
        try:
            embeddings = generate_embeddings(
                text_chunks,
                model_name=config.MODEL_NAME, # Ensure consistency
                batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to generate embeddings for chunks from {doc_path.name}: {e}", exc_info=True)
            continue # Skip to next document

        if not embeddings or len(embeddings) != len(text_chunks):
            logger.error(f"Embedding generation failed or produced mismatched number of embeddings for {doc_path.name}. Expected {len(text_chunks)}, Got {len(embeddings) if embeddings else 0}.")
            continue

        # Add embeddings to vector store
        try:
            vector_store.add_embeddings(text_chunks, embeddings)
            total_chunks_added += len(text_chunks)
            logger.info(f"Successfully added {len(text_chunks)} chunks from {doc_path.name} to vector store.")
        except Exception as e:
            logger.error(f"Failed to add embeddings from {doc_path.name} to vector store: {e}", exc_info=True)
            # Depending on the error, might want to retry or handle partial adds if applicable

    # --- 5. Save Index ---
    if total_chunks_added > 0:
        logger.info(f"A total of {total_chunks_added} new chunks were added. Saving the updated vector store index...")
        try:
            vector_store.save_index()
            logger.info("Vector store index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save vector store index: {e}", exc_info=True)
    else:
        logger.info("No new chunks were added to the vector store. Index save not required.")

    logger.info("Document ingestion process completed.")
    logger.info(f"Total documents processed from input directory: {len(documents_map)}")
    logger.info(f"Total chunks added to the vector store in this run: {total_chunks_added}")
    logger.info(f"Vector store now contains a total of {vector_store.get_ntotal()} items.")

if __name__ == "__main__":
    main()

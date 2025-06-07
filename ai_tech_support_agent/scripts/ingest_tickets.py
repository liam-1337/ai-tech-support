import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the 'app' directory is in the Python path for module imports
# This is crucial for running scripts from a subdirectory if the project isn't installed as a package.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.logic.ticket_loader import load_and_parse_ticket_data, TicketData
from app.logic.ticket_analyzer import analyze_tickets_batch, ExtractedTicketInfo
from app.logic.draft_generator import generate_draft_document
from app.logic.text_processor import process_and_chunk_text
from app.logic.embedding_generator import EmbeddingModel, generate_embeddings
from app.logic.vector_store_manager import VectorStoreManager
from app.logic.llm_handler import LLMGenerator # To ensure LLM is initialized for analyzer
from app import config

# Logger will be configured in main() based on command-line args
logger = logging.getLogger(__name__)

def setup_logging(log_level_str: str):
    """Configures basic logging for the script."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info(f"Logging level set to {log_level_str.upper()}")


def main():
    """
    Main script to process an uploaded ticket data JSON file, generate draft documentation
    from it, and ingest this documentation into the RAG vector store.
    """
    parser = argparse.ArgumentParser(
        description="Process ticket data JSON, generate drafts, and ingest into RAG vector store."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the JSON ticket data file."
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="token_count", # Default defined in text_processor, but can be overridden
        choices=["token_count", "paragraph", "sentence"],
        help="Chunking strategy for the generated Markdown documents. Default: token_count."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384, # Default from text_processor
        help="Max tokens per chunk if using 'token_count' strategy. Default: 384."
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=50, # Default from text_processor
        help="Overlap tokens between chunks if using 'token_count' strategy. Default: 50."
    )
    parser.add_argument(
        "--sentences-per-chunk",
        type=int,
        default=5, # Default from text_processor
        help="Sentences per chunk if using 'sentence' strategy. Default: 5."
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
        help="If set, delete any existing FAISS index in the configured directory before adding new data. "
             "Warning: This affects the entire index shared by other ingestion scripts using the same path."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for the script. Default: INFO."
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger.info("Starting ticket ingestion and RAG processing pipeline...")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Chunk strategy: {args.chunk_strategy}, Max tokens: {args.max_tokens}, Overlap: {args.overlap_tokens}")
    logger.info(f"Embedding batch size: {args.batch_size}, Recreate index: {args.recreate_index}")

    # --- 1. Initialize Models and Managers ---
    try:
        logger.info(f"Initializing embedding model: {config.MODEL_NAME}...")
        EmbeddingModel.set_model_name(config.MODEL_NAME)
        EmbeddingModel.get_model() # Trigger loading
        logger.info(f"Embedding model '{EmbeddingModel._model_name}' loaded on device '{EmbeddingModel._device}'.")

        logger.info(f"Initializing LLM ({config.LLM_MODEL_NAME}) for ticket analysis...")
        # LLMGenerator.get_llm_resources() will be called by analyze_tickets_batch, which handles initialization.
        # Explicitly calling it here ensures it's loaded before batch if preferred, but not strictly necessary.
        LLMGenerator.get_llm_resources()
        logger.info(f"LLM '{LLMGenerator._model_name_loaded}' loaded on device '{LLMGenerator._device}' (model device: {LLMGenerator._model.device if LLMGenerator._model else 'N/A'}).")

    except Exception as e:
        logger.error(f"Fatal error during model initialization: {e}", exc_info=True)
        sys.exit(1) # Exit if core models can't load

    vector_store_path = config.VECTOR_STORE_INDEX_DIR
    logger.info(f"Initializing vector store at: {vector_store_path} with dimension {config.EMBEDDING_DIMENSION}")
    if args.recreate_index and vector_store_path.exists():
        logger.warning(f"Recreate Index flag is set. Removing existing index files from {vector_store_path}...")
        index_file = vector_store_path / "vector_store.faiss"
        meta_file = vector_store_path / "vector_store_meta.pkl"
        try:
            if index_file.exists(): os.remove(index_file)
            if meta_file.exists(): os.remove(meta_file)
            logger.info("Existing index files removed.")
        except OSError as e:
            logger.error(f"Error removing existing index files: {e}. Proceeding with caution.", exc_info=True)
            # If removal fails, VectorStoreManager might still load old data if it doesn't re-init properly.

    try:
        vector_store = VectorStoreManager(
            index_dir_path=vector_store_path,
            dimension=config.EMBEDDING_DIMENSION
        )
        if args.recreate_index: # Ensure a fresh index if requested
            vector_store._create_new_index(config.EMBEDDING_DIMENSION)
        logger.info(f"Vector store initialized. Current items: {vector_store.get_ntotal()}, Dimension: {vector_store.get_dimension()}")
    except Exception as e:
        logger.error(f"Fatal error initializing VectorStoreManager: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Load and Parse Tickets ---
    try:
        logger.info(f"Loading and parsing tickets from {args.input_file}...")
        parsed_tickets: List[TicketData] = load_and_parse_ticket_data(args.input_file)
        if not parsed_tickets:
            logger.info("No tickets were successfully parsed from the input file. Exiting.")
            sys.exit(0)
        logger.info(f"Successfully parsed {len(parsed_tickets)} tickets.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}. Exiting.")
        sys.exit(1)
    except ValueError as e: # Handles JSON errors or wrong format
        logger.error(f"Error processing input file {args.input_file}: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred loading tickets: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Analyze Tickets (Summarization using LLM) ---
    try:
        logger.info("Analyzing tickets to extract problem/solution summaries...")
        # analyze_tickets_batch already handles LLM initialization if not done prior
        analyzed_ticket_info_list: List[ExtractedTicketInfo] = analyze_tickets_batch(parsed_tickets)
        if not analyzed_ticket_info_list:
            logger.info("No tickets were successfully analyzed (e.g., all irrelevant status or summarization failed). Exiting.")
            sys.exit(0)
        logger.info(f"Successfully analyzed {len(analyzed_ticket_info_list)} tickets.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during ticket analysis: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Process Each Analyzed Ticket for RAG Ingestion ---
    total_chunks_added_to_store = 0
    markdown_docs_generated_count = 0
    logger.info(f"Processing {len(analyzed_ticket_info_list)} analyzed tickets for RAG ingestion...")

    for i, extracted_info in enumerate(analyzed_ticket_info_list):
        logger.debug(f"Processing analyzed ticket {i+1}/{len(analyzed_ticket_info_list)}: ID {extracted_info.ticket_id}")
        try:
            # Generate Markdown Document
            markdown_text = generate_draft_document(extracted_info)
            if not markdown_text.strip():
                logger.warning(f"Generated empty Markdown for ticket ID {extracted_info.ticket_id}. Skipping.")
                continue
            markdown_docs_generated_count += 1
            logger.debug(f"Generated Markdown for ticket ID {extracted_info.ticket_id} (length: {len(markdown_text)}).")

            # Chunk Markdown
            chunk_params = {
                'model_name': config.MODEL_NAME, # For tokenizer if token_count strategy
                'max_tokens_per_chunk': args.max_tokens,
                'overlap_tokens': args.overlap_tokens,
                'sentences_per_chunk': args.sentences_per_chunk
            }
            text_chunks = process_and_chunk_text(
                raw_text=markdown_text,
                strategy=args.chunk_strategy,
                **chunk_params
            )
            if not text_chunks:
                logger.warning(f"No chunks produced for Markdown from ticket ID {extracted_info.ticket_id}. Skipping.")
                continue
            logger.debug(f"Ticket ID {extracted_info.ticket_id}: Produced {len(text_chunks)} chunks from Markdown.")

            # Generate Embeddings for Chunks
            embeddings = generate_embeddings(
                text_chunks,
                model_name=config.MODEL_NAME, # Embedding model
                batch_size=args.batch_size
            )
            if not embeddings or len(embeddings) != len(text_chunks):
                logger.error(f"Embedding generation failed or mismatched for ticket ID {extracted_info.ticket_id}. Skipping.")
                continue

            # Add to Vector Store
            vector_store.add_embeddings(text_chunks, embeddings)
            total_chunks_added_to_store += len(text_chunks)
            logger.info(f"Ticket ID {extracted_info.ticket_id}: Added {len(text_chunks)} chunks to vector store.")

        except Exception as e:
            logger.error(f"Failed to process ticket ID {extracted_info.ticket_id} for RAG ingestion: {e}", exc_info=True)
            # Continue with the next ticket

    # --- 5. Save Index ---
    if total_chunks_added_to_store > 0:
        try:
            logger.info(f"Total of {total_chunks_added_to_store} new chunks added. Saving the updated vector store index...")
            vector_store.save_index()
            logger.info("Vector store index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save vector store index: {e}", exc_info=True)
    else:
        logger.info("No new chunks were added to the vector store in this run. Index save not required.")

    logger.info("--- Ticket Ingestion and RAG Processing Summary ---")
    logger.info(f"Total tickets parsed from input file: {len(parsed_tickets)}")
    logger.info(f"Total tickets successfully analyzed (summarized): {len(analyzed_ticket_info_list)}")
    logger.info(f"Total Markdown documents generated: {markdown_docs_generated_count}")
    logger.info(f"Total text chunks added to vector store in this run: {total_chunks_added_to_store}")
    logger.info(f"Vector store now contains a grand total of {vector_store.get_ntotal()} items.")
    logger.info("Pipeline completed.")

if __name__ == "__main__":
    main()

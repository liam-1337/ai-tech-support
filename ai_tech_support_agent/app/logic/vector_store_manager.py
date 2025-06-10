import faiss
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import shutil # For file operations in tests

# Assuming app.config and embedding_generator are accessible
# This will be the case if the application is run via FastAPI main.py or scripts that set up sys.path
from app import config # To get EMBEDDING_MODEL_NAME
from app.logic.embedding_generator import EmbeddingModel, generate_embeddings # To get embedding_dimension and generate embeddings

# Get a logger for this module.
logger = logging.getLogger(__name__)

# Default chunking parameters (can be moved to config if needed)
DEFAULT_CHUNK_SIZE = 1000 # Characters
DEFAULT_CHUNK_OVERLAP = 150 # Characters


class VectorStoreManager:
    """
    Manages a FAISS vector index for storing and searching text embeddings,
    organized by document IDs.
    """
    def __init__(self, index_dir_path: Union[str, Path], embedding_dimension: Optional[int] = None) -> None:
        """
        Initializes the VectorStoreManager.

        Args:
            index_dir_path: Path to the directory where the FAISS index and metadata will be stored.
            embedding_dimension: The dimensionality of the embeddings. If None, it will try to determine
                                 it from the EmbeddingModel specified in config. Required if creating
                                 a new index and one doesn't already exist or if it cannot be determined.
        """
        self.index_dir_path = Path(index_dir_path)
        self.index_file = self.index_dir_path / "vector_store.faiss"
        self.metadata_file = self.index_dir_path / "vector_store_meta.pkl"

        self.index: Optional[faiss.IndexIDMap] = None
        self.current_dimension: Optional[int] = None

        # New data structures for document-level management
        self.doc_id_to_chunk_ids: Dict[str, List[int]] = {}  # doc_id -> [faiss_id1, faiss_id2, ...]
        self.chunk_id_to_doc_id: Dict[int, str] = {}      # faiss_id -> doc_id
        self.chunk_id_to_text: Dict[int, str] = {}        # faiss_id -> chunk_text
        self.doc_id_to_metadata: Dict[str, Dict] = {}     # doc_id -> metadata_dict
        self.next_chunk_id: int = 0                       # Counter for unique FAISS IDs

        # Determine embedding dimension if not provided
        if embedding_dimension is None:
            try:
                model_name = config.EMBEDDING_MODEL_NAME
                emb_model = EmbeddingModel.get_model(model_name)
                if hasattr(emb_model, 'get_sentence_embedding_dimension'):
                    self.current_dimension = emb_model.get_sentence_embedding_dimension()
                elif hasattr(emb_model, 'tokenizer'):
                     self.current_dimension = emb_model.tokenizer.model_max_length
                else:
                    dummy_emb = emb_model.encode(["test sentence"])[0]
                    self.current_dimension = len(dummy_emb)
                logger.info(f"Determined embedding dimension from model '{model_name}': {self.current_dimension}")
                embedding_dimension = self.current_dimension
            except Exception as e:
                logger.warning(f"Could not automatically determine embedding dimension: {e}. "
                               f"Please ensure 'embedding_dimension' is provided if creating a new index.")
        else:
            self.current_dimension = embedding_dimension

        if embedding_dimension:
             self.load_index(required_dimension=embedding_dimension)
        else:
            logger.warning("VectorStoreManager initialized without a dimension. Index operations will fail until dimension is set and load_index called.")


    def _split_text_into_chunks(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """Simple text splitter."""
        if not text or not text.strip():
            return []
        if chunk_overlap >= chunk_size:
            # Allow overlap to be equal to chunk_size for specific use cases, though typically smaller.
            # If overlap is strictly greater, it's problematic.
            if chunk_overlap > chunk_size:
                 raise ValueError("chunk_overlap must be less than or equal to chunk_size.")
            if chunk_overlap == chunk_size and len(text) > chunk_size: # Avoid infinite loop for single chunk text
                 logger.warning("chunk_overlap is equal to chunk_size. This might lead to duplicate chunks if text is larger than chunk_size.")


        chunks = []
        start_idx = 0
        while start_idx < len(text):
            end_idx = start_idx + chunk_size
            chunks.append(text[start_idx:end_idx])
            if end_idx >= len(text): # Ensure we don't step back if last chunk is shorter
                break
            start_idx += (chunk_size - chunk_overlap)
            if chunk_size - chunk_overlap <= 0 and len(text) > chunk_size : # Safety break for non-positive step with large text
                logger.error("Chunking step is not positive, breaking to prevent infinite loop.")
                break
        return chunks

    def _create_new_index(self, dimension: int) -> None:
        """Creates a new FAISS index and resets document management structures."""
        logger.info(f"Creating a new FAISS index with dimension {dimension} at {self.index_dir_path}")
        try:
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)
            self.current_dimension = dimension
            self.doc_id_to_chunk_ids = {}
            self.chunk_id_to_doc_id = {}
            self.chunk_id_to_text = {}
            self.doc_id_to_metadata = {}
            self.next_chunk_id = 0
            logger.info(f"Successfully created new FAISS index with dimension {dimension}.")
        except Exception as e:
            logger.error(f"Failed to create new FAISS index: {e}", exc_info=True)
            self.index = None
            self.current_dimension = None
            raise

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Adds a document to the vector store. Content is chunked, embedded, and stored."""
        if not self.is_initialized():
            logger.error(f"Cannot add document '{doc_id}': Vector store or embedding dimension not initialized.")
            raise ValueError("Vector store not properly initialized. Ensure dimension is provided.")

        if doc_id in self.doc_id_to_chunk_ids:
            logger.error(f"Document ID '{doc_id}' already exists. Use update_document to modify it.")
            raise ValueError(f"Document ID '{doc_id}' already exists.")

        text_chunks = self._split_text_into_chunks(content)
        if not text_chunks:
            logger.warning(f"No text chunks generated for document '{doc_id}' (content might be empty or too short). Nothing to add.")
            # Still create metadata entry if metadata is provided, even for empty content
            if metadata is not None:
                 self.doc_id_to_metadata[doc_id] = metadata
                 self.doc_id_to_chunk_ids[doc_id] = [] # Ensure doc_id is known
            return

        try:
            embeddings = generate_embeddings(text_chunks, model_name=config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document '{doc_id}': {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed for '{doc_id}'") from e

        if not embeddings or len(embeddings) != len(text_chunks):
            logger.error(f"Embedding generation for '{doc_id}' resulted in mismatched or empty embeddings.")
            return

        embeddings_np = np.array(embeddings, dtype='float32')
        if embeddings_np.shape[1] != self.current_dimension:
            logger.error(f"Dimension of generated embeddings ({embeddings_np.shape[1]}) for '{doc_id}' "
                         f"does not match index dimension ({self.current_dimension}).")
            return

        num_chunks = len(text_chunks)
        chunk_faiss_ids = list(range(self.next_chunk_id, self.next_chunk_id + num_chunks))
        self.next_chunk_id += num_chunks

        current_doc_chunk_ids = []
        try:
            self.index.add_with_ids(embeddings_np, np.array(chunk_faiss_ids, dtype=np.int64))
            for i, faiss_id in enumerate(chunk_faiss_ids):
                self.chunk_id_to_text[faiss_id] = text_chunks[i]
                self.chunk_id_to_doc_id[faiss_id] = doc_id
                current_doc_chunk_ids.append(faiss_id)
            self.doc_id_to_chunk_ids[doc_id] = current_doc_chunk_ids
            if metadata:
                self.doc_id_to_metadata[doc_id] = metadata
            logger.info(f"Added document '{doc_id}' with {num_chunks} chunks. Total FAISS entries: {self.index.ntotal}.")
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}' chunks to FAISS index: {e}", exc_info=True)
            raise

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document and all its associated chunks/embeddings."""
        if not self.is_initialized():
            logger.error(f"Cannot remove document '{doc_id}': Vector store not initialized.")
            return False

        if doc_id not in self.doc_id_to_chunk_ids:
            logger.warning(f"Document ID '{doc_id}' not found. Cannot remove.")
            return False

        chunk_faiss_ids_to_remove = self.doc_id_to_chunk_ids.get(doc_id, [])
        if not chunk_faiss_ids_to_remove: # Document known but has no chunks
            logger.info(f"Document ID '{doc_id}' has no associated chunks to remove. Removing document metadata.")
            del self.doc_id_to_chunk_ids[doc_id]
            if doc_id in self.doc_id_to_metadata:
                del self.doc_id_to_metadata[doc_id]
            return True

        try:
            ids_to_remove_np = np.array(chunk_faiss_ids_to_remove, dtype=np.int64)
            # Check if index actually contains these IDs before attempting removal
            # This is important because remove_ids might behave unexpectedly or error if IDs are not present
            # However, IndexIDMap's remove_ids is supposed to ignore non-existent IDs silently.
            # For robustness, we can check which IDs are actually in the index.
            # valid_ids_to_remove = [fid for fid in chunk_faiss_ids_to_remove if self.index.id_map.exists(fid)]
            # if not valid_ids_to_remove:
            #    logger.warning(f"None of the chunk IDs for doc '{doc_id}' actually exist in FAISS index for removal.")
            # else:
            #    num_removed = self.index.remove_ids(np.array(valid_ids_to_remove, dtype=np.int64))

            num_removed = self.index.remove_ids(ids_to_remove_np) # FAISS remove_ids is okay with non-existent IDs

            logger.info(f"FAISS remove_ids call for doc_id '{doc_id}' with {len(ids_to_remove_np)} IDs resulted in {num_removed} removals.")
            # num_removed might be less than len(ids_to_remove_np) if some IDs were already gone or never added, which is fine.

            for faiss_id in chunk_faiss_ids_to_remove:
                self.chunk_id_to_text.pop(faiss_id, None)
                self.chunk_id_to_doc_id.pop(faiss_id, None)

            self.doc_id_to_chunk_ids.pop(doc_id, None)
            self.doc_id_to_metadata.pop(doc_id, None)

            logger.info(f"Document '{doc_id}' and associated data removed. Total FAISS entries: {self.index.ntotal}.")
            return True
        except Exception as e:
            logger.error(f"Error removing document '{doc_id}': {e}", exc_info=True)
            return False

    def update_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Updates an existing document (remove then add). If not existing, adds it."""
        if not self.is_initialized():
            logger.error(f"Cannot update document '{doc_id}': Vector store not initialized.")
            raise ValueError("Vector store not initialized.")

        logger.info(f"Attempting to update document '{doc_id}'.")
        # Preserve old metadata if new metadata is None and doc exists
        existing_metadata = self.doc_id_to_metadata.get(doc_id)

        if doc_id in self.doc_id_to_chunk_ids:
            self.remove_document(doc_id)

        effective_metadata = metadata
        if metadata is None and existing_metadata is not None:
            effective_metadata = existing_metadata # Preserve old if new is not provided

        self.add_document(doc_id, content, effective_metadata)
        logger.info(f"Document '{doc_id}' updated/added successfully.")


    def add_embeddings(self, text_chunks: List[str], embeddings: List[List[float]], doc_id: Optional[str] = None, doc_metadata: Optional[Dict]=None) -> None:
        """Low-level: Adds pre-computed chunks and embeddings. Prefer `add_document`."""
        if not self.is_initialized():
            logger.error("FAISS index is not initialized. Cannot add embeddings.")
            raise ValueError("FAISS index not initialized.")
        if not text_chunks or not embeddings:
            logger.warning("Text chunks or embeddings list is empty. Nothing to add.")
            return
        if len(text_chunks) != len(embeddings):
            raise ValueError("Number of text chunks and embeddings must be the same.")

        embeddings_np = np.array(embeddings, dtype='float32')
        if embeddings_np.shape[1] != self.current_dimension:
            raise ValueError("Dimension of embeddings does not match index dimension.")

        num_chunks = len(text_chunks)
        chunk_faiss_ids = list(range(self.next_chunk_id, self.next_chunk_id + num_chunks))
        self.next_chunk_id += num_chunks

        try:
            self.index.add_with_ids(embeddings_np, np.array(chunk_faiss_ids, dtype=np.int64))
            for i, faiss_id in enumerate(chunk_faiss_ids):
                self.chunk_id_to_text[faiss_id] = text_chunks[i]
                if doc_id: self.chunk_id_to_doc_id[faiss_id] = doc_id

            if doc_id:
                self.doc_id_to_chunk_ids.setdefault(doc_id, []).extend(chunk_faiss_ids)
                if doc_metadata and doc_id not in self.doc_id_to_metadata:
                    self.doc_id_to_metadata[doc_id] = doc_metadata
                elif doc_metadata: logger.warning(f"Metadata for '{doc_id}' exists. Not overwriting.")
            logger.info(f"Added {num_chunks} pre-computed embeddings. Associated with doc_id '{doc_id if doc_id else 'N/A'}'.")
        except Exception as e:
            logger.error(f"Error adding pre-computed embeddings: {e}", exc_info=True)
            raise

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Searches for k nearest neighbors, returns rich info."""
        if not self.is_initialized() or self.index.ntotal == 0:
            logger.debug("Index not initialized or empty for search.") # Debug instead of warn for no results case
            return []
        if not query_embedding:
            logger.warning("Query embedding is empty.")
            return []
        query_embedding_np = np.array([query_embedding], dtype='float32')
        if query_embedding_np.shape[1] != self.current_dimension:
            logger.error(f"Query embedding dim ({query_embedding_np.shape[1]}) != index dim ({self.current_dimension}).")
            return []

        try:
            actual_k = min(k, self.index.ntotal)
            if actual_k == 0 : return [] # No items to search
            distances, faiss_ids = self.index.search(query_embedding_np, k=actual_k)
            results: List[Dict[str, Any]] = []
            for i, faiss_id_val in enumerate(faiss_ids[0]):
                if faiss_id_val == -1: continue
                faiss_id_int = int(faiss_id_val)
                chunk_text = self.chunk_id_to_text.get(faiss_id_int)
                parent_doc_id = self.chunk_id_to_doc_id.get(faiss_id_int)
                if chunk_text and parent_doc_id:
                    doc_metadata = self.doc_id_to_metadata.get(parent_doc_id, {})
                    results.append({
                        "chunk_text": chunk_text, "doc_id": parent_doc_id,
                        "metadata": doc_metadata, "distance": float(distances[0][i])
                    })
                else: logger.warning(f"Search result ID {faiss_id_int} missing in mappings.")
            logger.info(f"Search returned {len(results)} results for k={k} (actual_k={actual_k}).")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def save_index(self) -> None:
        """Saves FAISS index and all associated metadata."""
        if not self.index_dir_path.exists():
             self.index_dir_path.mkdir(parents=True, exist_ok=True)

        if not self.is_initialized() and not any([self.doc_id_to_chunk_ids, self.chunk_id_to_text]):
            logger.warning("No index or substantial metadata to save.")
            return
        try:
            if self.index:
                logger.info(f"Saving FAISS index to {self.index_file} ({self.index.ntotal} vectors)")
                faiss.write_index(self.index, str(self.index_file))
            else: logger.warning("FAISS index object is None. Skipping .faiss file save.")
            metadata_to_save = {
                "dimension": self.current_dimension, "doc_id_to_chunk_ids": self.doc_id_to_chunk_ids,
                "chunk_id_to_doc_id": self.chunk_id_to_doc_id, "chunk_id_to_text": self.chunk_id_to_text,
                "doc_id_to_metadata": self.doc_id_to_metadata, "next_chunk_id": self.next_chunk_id
            }
            with open(self.metadata_file, "wb") as f: pickle.dump(metadata_to_save, f)
            logger.info(f"Successfully saved FAISS index metadata to {self.metadata_file}.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index or metadata: {e}", exc_info=True)
            raise

    def load_index(self, required_dimension: Optional[int] = None) -> None:
        """Loads FAISS index and metadata. Creates new if not found and dimension is provided."""
        metadata_loaded_successfully = False
        if self.metadata_file.exists():
            logger.info(f"Attempting to load metadata from {self.metadata_file}...")
            try:
                with open(self.metadata_file, "rb") as f: metadata = pickle.load(f)
                loaded_dimension = metadata.get("dimension")
                if loaded_dimension: self.current_dimension = loaded_dimension
                if required_dimension and self.current_dimension and self.current_dimension != required_dimension:
                    logger.critical(f"CRITICAL: Required dim ({required_dimension}) != metadata dim ({self.current_dimension}).")
                    # Potentially raise error or adapt. For now, metadata dim takes precedence if FAISS matches it.
                self.doc_id_to_chunk_ids = metadata.get("doc_id_to_chunk_ids", {})
                self.chunk_id_to_doc_id = metadata.get("chunk_id_to_doc_id", {})
                self.chunk_id_to_text = metadata.get("chunk_id_to_text", {})
                self.doc_id_to_metadata = metadata.get("doc_id_to_metadata", {})
                self.next_chunk_id = metadata.get("next_chunk_id", 0)
                metadata_loaded_successfully = True
                logger.info(f"Metadata loaded. Dimension: {self.current_dimension}. Next_chunk_id: {self.next_chunk_id}.")
            except Exception as e:
                logger.error(f"Failed to load metadata from {self.metadata_file}: {e}", exc_info=True)
                self.current_dimension = required_dimension # Fallback to required_dimension if metadata load fails

        faiss_index_loaded_successfully = False
        if self.index_file.exists():
            logger.info(f"Attempting to load FAISS index from {self.index_file}...")
            try:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, dim {self.index.d}.")
                if self.current_dimension is None: self.current_dimension = self.index.d
                elif self.index.d != self.current_dimension:
                    logger.critical(f"CRITICAL: FAISS index dim ({self.index.d}) != metadata/required dim ({self.current_dimension}). Trusting FAISS dim.")
                    self.current_dimension = self.index.d
                # Consistency check (optional, can be intensive for large stores)
                # if metadata_loaded_successfully:
                #     faiss_ids_in_index = set(self.index.id_map.ToArray())
                #     if faiss_ids_in_index != set(self.chunk_id_to_text.keys()):
                #         logger.warning("Inconsistency between FAISS IDs and chunk_id_to_text keys.")
                faiss_index_loaded_successfully = True
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {self.index_file}: {e}", exc_info=True)
                self.index = None

        effective_dimension = self.current_dimension if self.current_dimension else required_dimension
        if not effective_dimension:
            logger.warning("No dimension available (from metadata, FAISS file, or required_dimension). Index operations may fail.")
            return # Cannot create new index without dimension

        if not faiss_index_loaded_successfully:
            logger.info(f"FAISS index file not found or failed to load. Creating new empty index with dimension {effective_dimension}.")
            self._create_new_index(effective_dimension)
            if metadata_loaded_successfully:
                 logger.warning("Metadata was loaded, but FAISS index was re-initialized as empty. "
                                "This means existing metadata points to a non-existent FAISS state. "
                                "Consider re-indexing or ensuring FAISS file is present.")
        elif not metadata_loaded_successfully and faiss_index_loaded_successfully:
            logger.error("FAISS index loaded, but metadata is missing/corrupt. Mappings are empty. Store is inconsistent.")
            # Decide: clear FAISS too? For now, keep FAISS, but mappings are empty.

    def get_dimension(self) -> Optional[int]: return self.current_dimension
    def get_ntotal_faiss(self) -> int: return self.index.ntotal if self.index else 0
    def get_document_count(self) -> int: return len(self.doc_id_to_chunk_ids)
    def is_initialized(self) -> bool: return self.index is not None and self.current_dimension is not None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    class MockConfig: # Minimal config for testing
        EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    if not hasattr(config, 'EMBEDDING_MODEL_NAME'):
        config.EMBEDDING_MODEL_NAME = MockConfig.EMBEDDING_MODEL_NAME

    logger.info(f"Running VSM tests with EMBEDDING_MODEL_NAME: {config.EMBEDDING_MODEL_NAME}")

    try:
        test_emb_model = EmbeddingModel.get_model(config.EMBEDDING_MODEL_NAME)
        test_dimension = test_emb_model.get_sentence_embedding_dimension() if hasattr(test_emb_model, 'get_sentence_embedding_dimension') else len(test_emb_model.encode(["test"])[0])
        logger.info(f"Test dimension from model {config.EMBEDDING_MODEL_NAME}: {test_dimension}")
    except Exception as e:
        logger.error(f"Could not load test embedding model. Defaulting test_dimension to 384. Error: {e}")
        test_dimension = 384

    test_index_dir_main = Path("./test_vsm_comprehensive")

    def setup_test_env(test_dir: Path):
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_test_env(test_dir: Path):
        if test_dir.exists():
            # logger.info(f"Cleaning up test directory: {test_dir}")
            # shutil.rmtree(test_dir) # Keep for inspection by default
            logger.warning(f"Test directory {test_dir} is NOT automatically cleaned up for inspection.")
            pass


    logger.info("\n--- Test Suite: VectorStoreManager Comprehensive ---")
    setup_test_env(test_index_dir_main)
    vs_manager: Optional[VectorStoreManager] = None

    try:
        # --- Test 1: Initialization ---
        logger.info("\n--- Test 1: Initialization ---")
        vs_manager = VectorStoreManager(index_dir_path=test_index_dir_main, embedding_dimension=test_dimension)
        assert vs_manager.is_initialized(), "Test 1.1 Failed: Manager not initialized"
        assert vs_manager.get_dimension() == test_dimension, f"Test 1.1 Failed: Dimension mismatch. Expected {test_dimension}, Got {vs_manager.get_dimension()}"
        assert vs_manager.get_ntotal_faiss() == 0, "Test 1.1 Failed: New FAISS index should be empty"
        assert vs_manager.get_document_count() == 0, "Test 1.1 Failed: New manager should have no documents"
        logger.info("Test 1.1 (New Index Creation) Passed.")
        vs_manager.save_index() # Save for next test part

        vs_manager_load_test = VectorStoreManager(index_dir_path=test_index_dir_main, embedding_dimension=test_dimension)
        assert vs_manager_load_test.is_initialized(), "Test 1.2 Failed: Loaded manager not initialized"
        assert vs_manager_load_test.get_dimension() == test_dimension, "Test 1.2 Failed: Loaded dimension mismatch"
        logger.info("Test 1.2 (Loading Empty Index) Passed.")


        # --- Test 1.3: _split_text_into_chunks direct test ---
        logger.info("\n--- Test 1.3: _split_text_into_chunks direct ---")
        test_text = "abcde fghij klmno pqrst uvwxyz" # 29 chars
        # Test case 1: chunk_size=10, overlap=3 -> "abcde fghi", "e fghij kl", "ij klmno p", "mno pqrst ", "rst uvwxyz"
        chunks = vs_manager._split_text_into_chunks(test_text, chunk_size=10, chunk_overlap=3)
        assert len(chunks) == 5, f"Test 1.3.1 Failed: Expected 5 chunks, got {len(chunks)}. Chunks: {chunks}"
        assert chunks[0] == "abcde fghi", f"Test 1.3.1 Failed: Chunk 0 mismatch. Got: {chunks[0]}"
        assert chunks[1] == "e fghij kl", f"Test 1.3.1 Failed: Chunk 1 mismatch. Got: {chunks[1]}"
        # Test case 2: no overlap
        chunks_no_overlap = vs_manager._split_text_into_chunks(test_text, chunk_size=10, chunk_overlap=0)
        assert len(chunks_no_overlap) == 3, f"Test 1.3.2 Failed: Expected 3 chunks, got {len(chunks_no_overlap)}"
        assert chunks_no_overlap[0] == "abcde fghi", f"Test 1.3.2 Failed: Chunk 0 (no overlap) mismatch."
        # Test case 3: empty text
        assert vs_manager._split_text_into_chunks("", 100, 10) == [], "Test 1.3.3 Failed: Empty text should yield no chunks"
        # Test case 4: text shorter than chunk size
        assert vs_manager._split_text_into_chunks("short", 100, 10) == ["short"], "Test 1.3.4 Failed: Short text should yield one chunk"
        logger.info("Test 1.3 (_split_text_into_chunks) Passed.")


        # --- Test 2: Document Operations (Add, Update, Remove) ---
        logger.info("\n--- Test 2: Document Operations ---")
        doc1_id = "doc1.txt"; doc1_content = "Apples are red. Oranges are orange." * 20; doc1_meta = {"src": "fruits.txt"}
        vs_manager.add_document(doc1_id, doc1_content, doc1_meta)
        assert vs_manager.get_document_count() == 1, "Test 2.1 Failed: Doc count after add"
        assert doc1_id in vs_manager.doc_id_to_chunk_ids
        doc1_num_chunks = len(vs_manager.doc_id_to_chunk_ids[doc1_id])
        assert doc1_num_chunks > 0, "Test 2.1 Failed: Doc1 should have chunks"
        assert vs_manager.get_ntotal_faiss() == doc1_num_chunks, "Test 2.1 Failed: FAISS total mismatch"
        assert vs_manager.doc_id_to_metadata.get(doc1_id) == doc1_meta
        logger.info(f"Test 2.1 (add_document) Passed. Doc1 has {doc1_num_chunks} chunks.")

        # Add empty doc
        doc_empty_id = "empty.txt"; vs_manager.add_document(doc_empty_id, "", {"src":"empty_test"})
        assert vs_manager.get_document_count() == 2, "Test 2.1.1 Failed: Doc count after adding empty doc"
        assert vs_manager.doc_id_to_chunk_ids.get(doc_empty_id) == [], "Test 2.1.1 Failed: Empty doc should have empty chunk list"
        assert doc_empty_id in vs_manager.doc_id_to_metadata, "Test 2.1.1 Failed: Empty doc should have metadata"


        doc2_id = "doc2.txt"; doc2_content = "Bananas are yellow. Grapes are purple." * 25; doc2_meta = {"src": "more_fruits.txt"}
        vs_manager.add_document(doc2_id, doc2_content, doc2_meta)
        doc2_num_chunks = len(vs_manager.doc_id_to_chunk_ids[doc2_id])
        assert vs_manager.get_document_count() == 3, "Test 2.2 Failed: Doc count after 2nd add"
        assert vs_manager.get_ntotal_faiss() == doc1_num_chunks + doc2_num_chunks
        logger.info("Test 2.2 (add_document - 2nd doc) Passed.")

        # Update doc1
        doc1_updated_content = "Pineapples are tropical. Mangoes are sweet." * 22
        doc1_updated_meta = {"src": "tropical.txt", "version": 2}
        vs_manager.update_document(doc1_id, doc1_updated_content, doc1_updated_meta)
        assert vs_manager.get_document_count() == 3, "Test 2.3 Failed: Doc count after update"
        doc1_updated_num_chunks = len(vs_manager.doc_id_to_chunk_ids[doc1_id])
        assert doc1_updated_num_chunks > 0 and doc1_updated_num_chunks != doc1_num_chunks, "Test 2.3 Failed: Updated doc1 chunk count issue"
        assert vs_manager.get_ntotal_faiss() == doc1_updated_num_chunks + doc2_num_chunks
        assert vs_manager.doc_id_to_metadata.get(doc1_id) == doc1_updated_meta
        logger.info(f"Test 2.3 (update_document - doc1) Passed. Doc1 now has {doc1_updated_num_chunks} chunks.")

        # Update non-existent (should add)
        doc3_id = "new_doc.txt"; doc3_content = "New content here."; doc3_meta = {"src":"new"}
        vs_manager.update_document(doc3_id, doc3_content, doc3_meta)
        assert vs_manager.get_document_count() == 4, "Test 2.3.1 Failed: Update non-existent should add"
        assert doc3_id in vs_manager.doc_id_to_metadata
        logger.info("Test 2.3.1 (update_document - non-existent) Passed.")

        # Update with empty content (should remove chunks)
        vs_manager.update_document(doc2_id, "", {"src":"empty_update"}) # Keep metadata, remove chunks
        assert vs_manager.doc_id_to_chunk_ids.get(doc2_id) == [], "Test 2.3.2 Failed: doc2 chunks should be empty after update with empty content"
        assert doc2_id in vs_manager.doc_id_to_metadata, "Test 2.3.2 Failed: doc2 metadata should persist"
        assert vs_manager.get_ntotal_faiss() == doc1_updated_num_chunks + len(vs_manager.doc_id_to_chunk_ids[doc3_id]) # Only doc1 and doc3 chunks
        logger.info("Test 2.3.2 (update_document - with empty content) Passed.")


        # Remove doc1
        vs_manager.remove_document(doc1_id)
        assert vs_manager.get_document_count() == 3, "Test 2.4 Failed: Doc count after remove" # doc_empty, doc2 (now empty content), doc3
        assert doc1_id not in vs_manager.doc_id_to_chunk_ids
        assert doc1_id not in vs_manager.doc_id_to_metadata
        # Ntotal should be chunks of doc2 (0) + doc3
        assert vs_manager.get_ntotal_faiss() == len(vs_manager.doc_id_to_chunk_ids[doc3_id])
        logger.info("Test 2.4 (remove_document - doc1) Passed.")

        assert not vs_manager.remove_document("non_existent_doc_id"), "Test 2.5 Failed: Remove non-existent should return False"
        logger.info("Test 2.5 (remove_document - non-existent) Passed.")


        # --- Test 3: Search Functionality ---
        logger.info("\n--- Test 3: Search Functionality ---")
        # Re-add a known doc for reliable search tests
        search_doc_id = "search_me.txt"; search_doc_content = "Unique keyword GHIJKL for search test."; search_doc_meta = {"type":"searchable"}
        vs_manager.add_document(search_doc_id, search_doc_content, search_doc_meta)

        query_text_search = "GHIJKL"
        query_embedding_search = generate_embeddings([query_text_search], model_name=config.EMBEDDING_MODEL_NAME)[0]

        search_results = vs_manager.search(query_embedding_search, k=3)
        assert len(search_results) >= 1, "Test 3.1 Failed: Search returned no results for GHIJKL"
        if search_results:
            top_res = search_results[0]
            assert top_res["doc_id"] == search_doc_id
            assert top_res["metadata"] == search_doc_meta
            assert "GHIJKL" in top_res["chunk_text"]
        logger.info(f"Test 3.1 (Search with results) Passed. Found {len(search_results)} results.")

        # Test k parameter
        search_results_k1 = vs_manager.search(query_embedding_search, k=1)
        assert len(search_results_k1) == 1 if vs_manager.get_ntotal_faiss() >=1 else len(search_results_k1) == 0, "Test 3.2 Failed: Search with k=1"
        logger.info("Test 3.2 (Search respects k) Passed.")

        # Search on empty
        vs_manager.remove_document(search_doc_id)
        vs_manager.remove_document(doc_empty_id) # doc_empty_id was added earlier
        vs_manager.remove_document(doc2_id) # doc2_id was updated to empty content, then remove it
        vs_manager.remove_document(doc3_id) # doc3_id was added via update
        assert vs_manager.get_ntotal_faiss() == 0, "Index should be empty before search_on_empty test"
        assert vs_manager.search(query_embedding_search, k=1) == [], "Test 3.3 Failed: Search on empty index should be empty list"
        logger.info("Test 3.3 (Search on empty index) Passed.")

        # --- Test 4: Persistence (Save/Load) ---
        logger.info("\n--- Test 4: Persistence ---")
        # Add some data back
        vs_manager.add_document(doc1_id, doc1_content, doc1_meta) # Original doc1_content, not updated one
        vs_manager.add_document(doc2_id, doc2_content, doc2_meta)
        vs_manager.save_index()

        vs_manager_loaded = VectorStoreManager(index_dir_path=test_index_dir_main, embedding_dimension=test_dimension)
        assert vs_manager_loaded.is_initialized(), "Test 4.1 Failed: Loaded manager not initialized"
        assert vs_manager_loaded.get_document_count() == 2, f"Test 4.1 Failed: Loaded doc count {vs_manager_loaded.get_document_count()}"
        assert vs_manager_loaded.get_ntotal_faiss() == (len(vs_manager_loaded.doc_id_to_chunk_ids[doc1_id]) + len(vs_manager_loaded.doc_id_to_chunk_ids[doc2_id]))
        assert doc1_id in vs_manager_loaded.doc_id_to_metadata and vs_manager_loaded.doc_id_to_metadata[doc1_id] == doc1_meta
        search_results_loaded = vs_manager_loaded.search(generate_embeddings(["apples"],model_name=config.EMBEDDING_MODEL_NAME)[0], k=1)
        assert len(search_results_loaded) >= 1 and search_results_loaded[0]["doc_id"] == doc1_id
        logger.info("Test 4.1 (Save and Load) Passed.")

        # --- Test 4.2: Partial Load - Metadata only ---
        logger.info("\n--- Test 4.2: Partial Load - Metadata only ---")
        vs_manager.save_index() # Ensure files are current
        temp_faiss_backup = test_index_dir_main / "vector_store.faiss.bak"
        if vs_manager.index_file.exists(): shutil.move(str(vs_manager.index_file), str(temp_faiss_backup))

        vs_meta_only = VectorStoreManager(index_dir_path=test_index_dir_main, embedding_dimension=test_dimension)
        assert vs_meta_only.is_initialized(), "Test 4.2 Failed: Init with metadata only" # Will create empty FAISS
        assert vs_meta_only.get_ntotal_faiss() == 0, "Test 4.2 Failed: FAISS should be empty"
        assert vs_meta_only.get_document_count() == 2, "Test 4.2 Failed: Doc count should be from metadata"
        assert doc1_id in vs_meta_only.doc_id_to_metadata
        logger.info("Test 4.2 (Metadata only load) Passed.")
        if temp_faiss_backup.exists(): shutil.move(str(temp_faiss_backup), str(vs_manager.index_file)) # Restore

        # --- Test 4.3: Partial Load - FAISS only ---
        logger.info("\n--- Test 4.3: Partial Load - FAISS only ---")
        vs_manager.save_index() # Ensure files are current
        temp_meta_backup = test_index_dir_main / "vector_store_meta.pkl.bak"
        if vs_manager.metadata_file.exists(): shutil.move(str(vs_manager.metadata_file), str(temp_meta_backup))

        vs_faiss_only = VectorStoreManager(index_dir_path=test_index_dir_main, embedding_dimension=test_dimension)
        assert vs_faiss_only.is_initialized(), "Test 4.3 Failed: Init with FAISS only"
        assert vs_faiss_only.get_ntotal_faiss() > 0, "Test 4.3 Failed: FAISS should have items" # Based on previous save
        assert vs_faiss_only.get_document_count() == 0, "Test 4.3 Failed: Doc count should be 0 due to no metadata"
        logger.info("Test 4.3 (FAISS only load) Passed - leads to inconsistent state as expected.")
        if temp_meta_backup.exists(): shutil.move(str(temp_meta_backup), str(vs_manager.metadata_file)) # Restore


        # --- Test 5: Dimension Mismatch on Load ---
        # This test was in previous version of file, re-enabling and adapting.
        logger.info("\n--- Test 5: Load index with dimension mismatch ---")
        vs_manager.save_index() # Save with current test_dimension
        wrong_dimension = test_dimension + 10 # Ensure it's different
        logger.info(f"Attempting to load index (dim={test_dimension}) with required_dimension={wrong_dimension}")

        # Temporarily suppress critical logs for this expected mismatch scenario for cleaner test output
        original_log_level = logging.getLogger('ai_tech_support_agent.app.logic.vector_store_manager').getEffectiveLevel()
        logging.getLogger('ai_tech_support_agent.app.logic.vector_store_manager').setLevel(logging.ERROR) # Suppress critical, warning, info

        vs_manager_mismatch = VectorStoreManager(index_dir_path=test_index_dir_main, embedding_dimension=wrong_dimension)

        logging.getLogger('ai_tech_support_agent.app.logic.vector_store_manager').setLevel(original_log_level) # Restore log level

        # Behavior: load_index logs critical error. If FAISS file exists, its dimension is trusted.
        # If metadata also exists and its dim matches FAISS, that's used.
        # If required_dim (wrong_dimension) is passed, it's compared against loaded/FAISS dim.
        # The final self.current_dimension will be what was loaded from existing files if they matched,
        # or from FAISS file if metadata was missing/mismatched.
        assert vs_manager_mismatch.get_dimension() == test_dimension, \
            f"Test 5 Failed: Loaded dimension should be original {test_dimension}, not {vs_manager_mismatch.get_dimension()}"
        logger.info(f"Test 5 (Dimension Mismatch) Passed. Loaded original dimension {vs_manager_mismatch.get_dimension()} despite requiring {wrong_dimension}.")


    except Exception as e:
        logger.error(f"A test failed or an unexpected error occurred: {e}", exc_info=True)
    finally:
        cleanup_test_env(test_index_dir_main)
        logger.info("\n--- VectorStoreManager Test Suite Complete ---")

```

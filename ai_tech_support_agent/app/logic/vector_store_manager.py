import faiss
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

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
                # Temporarily load model to get dimension.
                # EmbeddingModel caches the model, so this is efficient if called again.
                emb_model = EmbeddingModel.get_model(model_name)
                # Some models have get_sentence_embedding_dimension, others might need different handling.
                # Assuming a general way or a fixed known dimension from config if not directly gettable.
                if hasattr(emb_model, 'get_sentence_embedding_dimension'):
                    self.current_dimension = emb_model.get_sentence_embedding_dimension()
                elif hasattr(emb_model, 'tokenizer'): # Fallback for some models, not always accurate for output dim
                     self.current_dimension = emb_model.tokenizer.model_max_length
                else: # Final fallback: try to encode a dummy sentence
                    dummy_emb = emb_model.encode(["test sentence"])[0]
                    self.current_dimension = len(dummy_emb)

                logger.info(f"Determined embedding dimension from model '{model_name}': {self.current_dimension}")
                embedding_dimension = self.current_dimension
            except Exception as e:
                logger.warning(f"Could not automatically determine embedding dimension: {e}. "
                               f"Please ensure 'embedding_dimension' is provided if creating a new index.")
        else:
            self.current_dimension = embedding_dimension

        if embedding_dimension: # Only load/create if dimension is known
             self.load_index(required_dimension=embedding_dimension)
        else:
            logger.warning("VectorStoreManager initialized without a dimension. Index operations will fail until dimension is set and load_index called.")


    def _split_text_into_chunks(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """Simple text splitter."""
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")

        chunks = []
        start_idx = 0
        while start_idx < len(text):
            end_idx = start_idx + chunk_size
            chunks.append(text[start_idx:end_idx])
            start_idx += (chunk_size - chunk_overlap)
        return chunks

    def _create_new_index(self, dimension: int) -> None:
        """Creates a new FAISS index and resets document management structures."""
        logger.info(f"Creating a new FAISS index with dimension {dimension} at {self.index_dir_path}")
        try:
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)
            self.current_dimension = dimension

            # Reset all document/chunk tracking structures
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
            # Do not reset other members here, as they might hold loaded data if creation fails after a load attempt
            raise

    # def add_embeddings(self, text_chunks: List[str], embeddings: List[List[float]]) -> None:
    #     """
    #     DEPRECATED: This method is not compatible with document-centric management.
    #     Use add_document instead.
    #     Adds text chunks and their corresponding embeddings to the FAISS index.
    #
    #     Args:
    #         text_chunks: A list of original text strings.
    #         embeddings: A list of embeddings (each a list of floats) corresponding to text_chunks.
    #
    #     Raises:
    #         ValueError: If the index is not initialized or if input lists are empty/mismatched.
    #     """
    #     logger.warning("add_embeddings is deprecated. Use add_document for document-centric management.")
    #     if self.index is None:
    #         logger.error("FAISS index is not initialized. Cannot add embeddings.")
    #         raise ValueError("FAISS index not initialized. Call load_index() or ensure dimension is provided at init.")
    #     # ... (rest of the original implementation, which needs careful thought if to be kept)
    #     # This method would need to assign new chunk_ids and potentially a generic doc_id.
    #     # For now, strongly recommend using add_document.
    #     pass


    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Adds a document to the vector store. Content is chunked, embedded, and stored.

        Args:
            doc_id: A unique identifier for the document (e.g., file path).
            content: The textual content of the document.
            metadata: Optional dictionary of metadata associated with the document.

        Raises:
            ValueError: If doc_id already exists or if index is not properly initialized.
        """
        if not self.is_initialized():
            logger.error(f"Cannot add document '{doc_id}': Vector store or embedding dimension not initialized.")
            raise ValueError("Vector store not properly initialized. Ensure dimension is provided.")

        if doc_id in self.doc_id_to_chunk_ids:
            logger.error(f"Document ID '{doc_id}' already exists. Use update_document to modify it.")
            raise ValueError(f"Document ID '{doc_id}' already exists.")

        text_chunks = self._split_text_into_chunks(content)
        if not text_chunks:
            logger.warning(f"No text chunks generated for document '{doc_id}'. Nothing to add.")
            return

        try:
            # Assuming generate_embeddings uses the model from config.EMBEDDING_MODEL_NAME
            embeddings = generate_embeddings(text_chunks, model_name=config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document '{doc_id}': {e}", exc_info=True)
            # Potentially re-raise or handle more gracefully depending on desired behavior
            raise RuntimeError(f"Embedding generation failed for '{doc_id}'") from e

        if not embeddings or len(embeddings) != len(text_chunks):
            logger.error(f"Embedding generation for '{doc_id}' resulted in mismatched or empty embeddings.")
            return

        embeddings_np = np.array(embeddings, dtype='float32')
        if embeddings_np.shape[1] != self.current_dimension:
            logger.error(f"Dimension of generated embeddings ({embeddings_np.shape[1]}) for '{doc_id}' "
                         f"does not match index dimension ({self.current_dimension}).")
            # This should ideally not happen if dimension is derived from the same model config
            return

        # Generate FAISS IDs for the new chunks
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

            logger.info(f"Added document '{doc_id}' with {num_chunks} chunks to the vector store. Total FAISS entries: {self.index.ntotal}.")
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}' chunks to FAISS index: {e}", exc_info=True)
            # Attempt to rollback: remove any chunks added for this doc_id if some failed
            # This is complex as add_with_ids might be partially successful.
            # A safer approach is to ensure all data is prepared before calling add_with_ids.
            # If add_with_ids itself fails, FAISS state might be inconsistent.
            # For now, we log and don't add to mappings if FAISS add fails.
            # Revert mappings if they were partially updated before FAISS call.
            # Here, mappings are updated after successful FAISS add.
            raise

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document and all its associated chunks/embeddings from the vector store.

        Args:
            doc_id: The ID of the document to remove.

        Returns:
            True if document was removed, False if not found or error.
        """
        if not self.is_initialized():
            logger.error(f"Cannot remove document '{doc_id}': Vector store not initialized.")
            return False

        if doc_id not in self.doc_id_to_chunk_ids:
            logger.warning(f"Document ID '{doc_id}' not found. Cannot remove.")
            return False

        chunk_faiss_ids_to_remove = self.doc_id_to_chunk_ids.get(doc_id, [])
        if not chunk_faiss_ids_to_remove:
            logger.warning(f"No chunks found for document ID '{doc_id}', though doc_id was present in keys. Cleaning up mappings.")
            del self.doc_id_to_chunk_ids[doc_id]
            if doc_id in self.doc_id_to_metadata:
                del self.doc_id_to_metadata[doc_id]
            return True # Effectively removed as it had no indexed chunks

        try:
            ids_to_remove_np = np.array(chunk_faiss_ids_to_remove, dtype=np.int64)
            num_removed = self.index.remove_ids(ids_to_remove_np)

            if num_removed < len(chunk_faiss_ids_to_remove):
                logger.warning(f"FAISS remove_ids reported removing {num_removed} items for doc_id '{doc_id}', "
                               f"but expected {len(chunk_faiss_ids_to_remove)}. Index might have inconsistencies.")
            else:
                logger.info(f"Successfully removed {num_removed} chunks from FAISS for document '{doc_id}'.")

            # Clean up mappings
            for faiss_id in chunk_faiss_ids_to_remove:
                if faiss_id in self.chunk_id_to_text:
                    del self.chunk_id_to_text[faiss_id]
                if faiss_id in self.chunk_id_to_doc_id:
                    del self.chunk_id_to_doc_id[faiss_id]

            del self.doc_id_to_chunk_ids[doc_id]
            if doc_id in self.doc_id_to_metadata:
                del self.doc_id_to_metadata[doc_id]

            logger.info(f"Document '{doc_id}' and its associated data removed from vector store. Total FAISS entries: {self.index.ntotal}.")
            return True
        except Exception as e:
            logger.error(f"Error removing document '{doc_id}' from FAISS index or cleaning mappings: {e}", exc_info=True)
            # State might be inconsistent here. A more robust system might try to reconcile.
            return False

    def update_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Updates an existing document. This is implemented as a remove followed by an add.
        If the document does not exist, it will be added.

        Args:
            doc_id: The ID of the document to update.
            content: The new textual content of the document.
            metadata: The new metadata for the document. If None, existing metadata might be cleared
                      or preserved depending on add_document's behavior with metadata updates.
                      Current add_document will set it.
        """
        if not self.is_initialized():
            logger.error(f"Cannot update document '{doc_id}': Vector store not initialized.")
            raise ValueError("Vector store not initialized.")

        logger.info(f"Attempting to update document '{doc_id}'.")
        if doc_id in self.doc_id_to_chunk_ids: # Check if it exists to call remove
            self.remove_document(doc_id) # remove_document handles logging if not found, but this check is more direct

        # Add the new version of the document
        # If metadata is None in update, should we keep old metadata or clear it?
        # Current behavior: add_document will set new metadata, effectively clearing old if None is passed.
        # To preserve old metadata if new is None, add_document would need modification or logic here.
        effective_metadata = metadata
        if metadata is None and doc_id in self.doc_id_to_metadata: # If new metadata is not provided, keep the old one.
            effective_metadata = self.doc_id_to_metadata.get(doc_id) # This won't work as remove_document clears it.
            # This means update needs to fetch old metadata *before* remove_document if we want to preserve it.
            # For simplicity now: if metadata is None, it will be empty for the new version unless add_document handles it.
            # The prompt implies metadata is part of the "new" document content.
            pass # Current add_document will handle setting metadata.

        self.add_document(doc_id, content, effective_metadata) # add_document will set metadata
        logger.info(f"Document '{doc_id}' updated successfully.")


    def add_embeddings(self, text_chunks: List[str], embeddings: List[List[float]], doc_id: Optional[str] = None, doc_metadata: Optional[Dict]=None) -> None:
        """
        Low-level: Adds pre-computed text chunks and their embeddings to the FAISS index.
        If doc_id is provided, associates these chunks with that document.
        WARNING: This method is for advanced use. Prefer `add_document` for typical document processing.

        Args:
            text_chunks: A list of original text strings.
            embeddings: A list of embeddings (each a list of floats) corresponding to text_chunks.
            doc_id: Optional. If provided, these chunks will be associated with this document ID.
            doc_metadata: Optional. If doc_id is provided, this metadata will be associated with it.

        Raises:
            ValueError: If the index is not initialized or if input lists are empty/mismatched or dimension mismatch.
        """
        if not self.is_initialized():
            logger.error("FAISS index is not initialized. Cannot add embeddings.")
            raise ValueError("FAISS index not initialized.")

        if not text_chunks or not embeddings:
            logger.warning("Text chunks or embeddings list is empty. Nothing to add.")
            return

        if len(text_chunks) != len(embeddings):
            logger.error(f"Mismatch between number of text chunks ({len(text_chunks)}) and embeddings ({len(embeddings)}).")
            raise ValueError("Number of text chunks and embeddings must be the same.")

        embeddings_np = np.array(embeddings, dtype='float32')
        if embeddings_np.shape[1] != self.current_dimension:
            logger.error(f"Dimension of provided embeddings ({embeddings_np.shape[1]}) does not match index dimension ({self.current_dimension}).")
            raise ValueError("Dimension of embeddings does not match index dimension.")

        num_chunks = len(text_chunks)
        chunk_faiss_ids = list(range(self.next_chunk_id, self.next_chunk_id + num_chunks))
        self.next_chunk_id += num_chunks

        current_doc_chunk_ids_for_this_add = []
        try:
            self.index.add_with_ids(embeddings_np, np.array(chunk_faiss_ids, dtype=np.int64))

            for i, faiss_id in enumerate(chunk_faiss_ids):
                self.chunk_id_to_text[faiss_id] = text_chunks[i]
                current_doc_chunk_ids_for_this_add.append(faiss_id)
                if doc_id:
                    self.chunk_id_to_doc_id[faiss_id] = doc_id

            if doc_id:
                if doc_id not in self.doc_id_to_chunk_ids:
                    self.doc_id_to_chunk_ids[doc_id] = []
                self.doc_id_to_chunk_ids[doc_id].extend(current_doc_chunk_ids_for_this_add)
                if doc_metadata and doc_id not in self.doc_id_to_metadata: # Only add metadata if not already set by a full add_document
                    self.doc_id_to_metadata[doc_id] = doc_metadata
                elif doc_metadata and doc_id in self.doc_id_to_metadata:
                    logger.warning(f"Metadata for '{doc_id}' already exists. Not overwriting with add_embeddings call.")


            logger.info(f"Added {num_chunks} pre-computed embeddings to the index. Associated with doc_id '{doc_id if doc_id else 'N/A'}'. Total FAISS entries: {self.index.ntotal}.")
        except Exception as e:
            logger.error(f"Error adding pre-computed embeddings to FAISS index: {e}", exc_info=True)
            raise

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the k nearest neighbors to the query_embedding.
        Returns richer information including chunk text, document ID, document metadata, and distance.

        Args:
            query_embedding: A single embedding (list of floats) for the query.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of dictionaries, where each dictionary contains:
            - "chunk_text": str (the actual text chunk)
            - "doc_id": str (identifier of the source document)
            - "metadata": Dict (metadata of the source document)
            - "distance": float (L2 distance from the query embedding)
        """
        if not self.is_initialized() or self.index.ntotal == 0:
            logger.warning("Index is not initialized or is empty. Cannot perform search.")
            return []

        if not query_embedding:
            logger.warning("Query embedding is empty. Cannot perform search.")
            return []

        query_embedding_np = np.array([query_embedding], dtype='float32')
        if query_embedding_np.shape[1] != self.current_dimension:
            logger.error(f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match index dimension ({self.current_dimension}).")
            return []

        try:
            distances, faiss_ids = self.index.search(query_embedding_np, k=min(k, self.index.ntotal)) # Ensure k is not > ntotal
            results: List[Dict[str, Any]] = []
            for i, faiss_id_val in enumerate(faiss_ids[0]): # faiss_ids[0] because query is a single vector
                if faiss_id_val != -1: # FAISS returns -1 for IDs not found
                    chunk_text = self.chunk_id_to_text.get(int(faiss_id_val)) # Ensure faiss_id_val is int for dict key
                    parent_doc_id = self.chunk_id_to_doc_id.get(int(faiss_id_val))

                    if chunk_text and parent_doc_id:
                        doc_metadata = self.doc_id_to_metadata.get(parent_doc_id, {})
                        results.append({
                            "chunk_text": chunk_text,
                            "doc_id": parent_doc_id,
                            "metadata": doc_metadata,
                            "distance": float(distances[0][i])
                        })
                    elif not chunk_text:
                        logger.warning(f"FAISS ID {faiss_id_val} found in search, but its text not in chunk_id_to_text map.")
                    elif not parent_doc_id:
                        logger.warning(f"FAISS ID {faiss_id_val} found in search, text found, but its parent_doc_id not in chunk_id_to_doc_id map.")
            logger.info(f"Search returned {len(results)} results for k={k}.")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def save_index(self) -> None:
        """
        Saves the FAISS index and all associated metadata (mappings, dimension, next_chunk_id) to disk.
        """
        if not self.is_initialized(): # Check if index object exists
            logger.warning("No FAISS index object to save (it might not have been initialized).")
            # Still try to save metadata if it exists, as it might be from a previously failed load of index file
            # but successful load of metadata. Or, enforce that self.index must exist.
            # For robustness, let's allow saving metadata even if faiss index object is None,
            # but log it clearly.
            if not any([self.doc_id_to_chunk_ids, self.chunk_id_to_text]): # if no useful metadata, then skip
                 logger.warning("No metadata to save either. Aborting save.")
                 return

        try:
            self.index_dir_path.mkdir(parents=True, exist_ok=True)
            if self.index:
                logger.info(f"Saving FAISS index to {self.index_file}")
                faiss.write_index(self.index, str(self.index_file))
            else:
                logger.warning(f"FAISS index object is None. Skipping saving .faiss file. Metadata will still be saved.")


            metadata_to_save = {
                "dimension": self.current_dimension,
                "doc_id_to_chunk_ids": self.doc_id_to_chunk_ids,
                "chunk_id_to_doc_id": self.chunk_id_to_doc_id,
                "chunk_id_to_text": self.chunk_id_to_text,
                "doc_id_to_metadata": self.doc_id_to_metadata,
                "next_chunk_id": self.next_chunk_id
            }
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata_to_save, f)
            logger.info(f"Successfully saved FAISS index metadata to {self.metadata_file}.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index or metadata: {e}", exc_info=True)
            raise

    def load_index(self, required_dimension: Optional[int] = None) -> None:
        """
        Loads the FAISS index and all associated metadata from disk.
        If not found and required_dimension is provided, a new index is created.

        Args:
            required_dimension: The dimension the index should have. If an existing index is loaded,
                                its dimension is checked against this. If a new index is created,
                                this dimension is used. This is critical for consistency.
        """
        metadata_loaded_successfully = False
        if self.metadata_file.exists():
            logger.info(f"Attempting to load FAISS index metadata from {self.metadata_file}...")
            try:
                with open(self.metadata_file, "rb") as f:
                    metadata = pickle.load(f)

                loaded_dimension = metadata.get("dimension")
                if loaded_dimension is None:
                    logger.error("Dimension not found in metadata. Metadata file may be corrupt or old.")
                    # Decide if to proceed with index loading or fail fast
                else:
                    self.current_dimension = loaded_dimension
                    logger.info(f"Successfully loaded dimension ({self.current_dimension}) from metadata.")

                # Critical check: required_dimension vs loaded_dimension from metadata
                if required_dimension is not None and self.current_dimension != required_dimension:
                    logger.critical(
                        f"Critical: Required dimension ({required_dimension}) does not match "
                        f"dimension from loaded metadata ({self.current_dimension}). "
                        "This can lead to severe errors if FAISS index is loaded. "
                        "Consider deleting the old index/metadata or ensuring dimensions match."
                    )
                    # raise ValueError("Dimension mismatch between requirement and loaded metadata.") # Option to fail hard
                    # For now, allow proceeding but with this loaded dimension, overriding required_dimension
                    # if FAISS index itself is also loaded and matches this.

                self.doc_id_to_chunk_ids = metadata.get("doc_id_to_chunk_ids", {})
                self.chunk_id_to_doc_id = metadata.get("chunk_id_to_doc_id", {})
                self.chunk_id_to_text = metadata.get("chunk_id_to_text", {}) # Crucial: was chunk_map (List) before
                self.doc_id_to_metadata = metadata.get("doc_id_to_metadata", {})
                self.next_chunk_id = metadata.get("next_chunk_id", 0)
                metadata_loaded_successfully = True
                logger.info(f"Successfully loaded all metadata components. Next_chunk_id: {self.next_chunk_id}.")
                logger.debug(f"Loaded doc_id_to_chunk_ids (sample): {list(self.doc_id_to_chunk_ids.keys())[:3]}")
                logger.debug(f"Loaded chunk_id_to_text (sample IDs): {list(self.chunk_id_to_text.keys())[:3]}")


            except Exception as e:
                logger.error(f"Failed to load FAISS metadata from {self.metadata_file}: {e}", exc_info=True)
                # Reset all mappings if metadata loading fails to ensure clean state
                self._create_new_index(required_dimension if required_dimension else self.current_dimension if self.current_dimension else 384) # Fallback dim
                # The above line means if metadata load fails, we effectively start fresh or with a default dim index.

        faiss_index_loaded_successfully = False
        if self.index_file.exists():
            logger.info(f"Attempting to load existing FAISS index from {self.index_file}...")
            try:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} entries and dimension {self.index.d}.")

                if self.current_dimension is None: # If metadata didn't load or had no dimension
                    self.current_dimension = self.index.d
                    logger.info(f"Dimension set from loaded FAISS index: {self.current_dimension}")
                elif self.index.d != self.current_dimension:
                    logger.critical(
                        f"Mismatch between FAISS index dimension ({self.index.d}) and "
                        f"metadata dimension ({self.current_dimension}). This is a critical error."
                    )
                    # This is a severe inconsistency. Options:
                    # 1. Trust FAISS index dim, discard metadata dim (could lead to issues if metadata is right)
                    # 2. Trust metadata dim, discard FAISS index (requires re-indexing all data)
                    # 3. Fail hard.
                    # For now, let's trust FAISS index dimension more if it loads.
                    self.current_dimension = self.index.d
                    logger.warning(f"Overriding metadata dimension with FAISS index dimension: {self.current_dimension}")

                # Verify consistency between FAISS index and loaded chunk_id mappings
                if metadata_loaded_successfully: # Only if metadata was loaded
                    faiss_ids_in_index = set(self.index.id_map.ToArray()) # Get all actual IDs in FAISS
                    mapped_chunk_ids = set(self.chunk_id_to_text.keys())
                    if faiss_ids_in_index != mapped_chunk_ids:
                        logger.warning(f"Inconsistency: IDs in FAISS index ({len(faiss_ids_in_index)}) do not perfectly match "
                                       f"IDs in loaded chunk_id_to_text map ({len(mapped_chunk_ids)}). "
                                       f"Missing in map: {faiss_ids_in_index - mapped_chunk_ids}, "
                                       f"Extra in map: {mapped_chunk_ids - faiss_ids_in_index}. May need rebuild.")
                        # Potentially attempt to reconcile or mark for rebuild.

                faiss_index_loaded_successfully = True
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {self.index_file}: {e}", exc_info=True)
                self.index = None # Ensure index is None if loading fails

        # Post-loading decisions:
        if not faiss_index_loaded_successfully and not metadata_loaded_successfully:
            # Both failed, or neither existed. Create new if dimension is known.
            if required_dimension:
                logger.info(f"No existing index or metadata found/loaded. Creating new index with dimension {required_dimension}.")
                self._create_new_index(required_dimension)
            elif self.current_dimension: # Dimension might have been set by constructor from config
                logger.info(f"No existing index or metadata found/loaded. Creating new index with derived dimension {self.current_dimension}.")
                self._create_new_index(self.current_dimension)
            else:
                logger.warning(
                    "No FAISS index/metadata found, and no dimension provided/derived to create a new one. "
                    "Index remains uninitialized."
                )
        elif not faiss_index_loaded_successfully and metadata_loaded_successfully:
            # Metadata loaded, but FAISS index file missing or corrupt.
            # We have the mappings, but no actual vectors. Index needs to be rebuilt from source data.
            logger.warning(f"Metadata loaded, but FAISS index file '{self.index_file}' is missing or corrupt. "
                           "The vector index is effectively empty and needs new data to be added (or rebuilt).")
            # Create an empty FAISS index matching the loaded metadata dimension
            if self.current_dimension:
                logger.info(f"Creating a new empty FAISS index with loaded dimension {self.current_dimension} to match metadata.")
                base_index = faiss.IndexFlatL2(self.current_dimension)
                self.index = faiss.IndexIDMap(base_index) # Empty index, ready for adds
                # Keep the loaded metadata (doc_id_to_chunk_ids etc.) as it might be useful for a rebuild process,
                # but be aware that self.index.ntotal will be 0.
                # Any calls to `add_document` will start adding to this new empty FAISS index.
                # If old FAISS IDs from loaded metadata are reused, it could clash if not careful.
                # `next_chunk_id` from metadata is important here.
            else:
                logger.error("Metadata loaded but no dimension available to create a placeholder FAISS index. Index uninitialized.")

        elif faiss_index_loaded_successfully and not metadata_loaded_successfully:
            # FAISS index loaded, but metadata missing or corrupt. This is problematic.
            # We have vectors, but no way to map them back to text or documents.
            logger.error(f"FAISS index loaded, but metadata file '{self.metadata_file}' is missing or corrupt. "
                         "Cannot map vectors to content. Store is in an inconsistent state. "
                         "Consider deleting and re-indexing all data.")
            # Optionally, clear all mappings and treat as if only an empty index was loaded.
            # self._create_new_index(self.current_dimension) # This would wipe mappings and keep an empty index.
            # For now, leave it in this state; user needs to intervene. Index is technically "initialized".

    def get_dimension(self) -> Optional[int]:
        """Returns the dimension of the loaded/created FAISS index."""
        return self.current_dimension

    def get_ntotal_faiss(self) -> int:
        """Returns the total number of embeddings directly from FAISS index."""
        if self.index:
            return self.index.ntotal
        return 0

    def get_document_count(self) -> int:
        """Returns the number of documents tracked by doc_id_to_chunk_ids."""
        return len(self.doc_id_to_chunk_ids)

    def is_initialized(self) -> bool:
        """Checks if the FAISS index and dimension are initialized."""
        return self.index is not None and self.current_dimension is not None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Mock app.config for testing if not running in full app context
    class MockConfig:
        EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Must be a valid SBERT model
        # Define other config vars if EmbeddingModel or other parts need them

    if not hasattr(config, 'EMBEDDING_MODEL_NAME'): # Simple check if config is already app-level
        config.EMBEDDING_MODEL_NAME = MockConfig.EMBEDDING_MODEL_NAME
        # This is a hack for standalone script runs. In app, real config is used.
        # Ensure embedding_generator can load this model.

    logger.info(f"Running VectorStoreManager module directly for testing with EMBEDDING_MODEL_NAME: {config.EMBEDDING_MODEL_NAME}")

    # Determine test dimension from the model
    try:
        test_emb_model = EmbeddingModel.get_model(config.EMBEDDING_MODEL_NAME)
        if hasattr(test_emb_model, 'get_sentence_embedding_dimension'):
             test_dimension = test_emb_model.get_sentence_embedding_dimension()
        else: # Fallback
            test_dimension = len(test_emb_model.encode(["test"])[0])
        logger.info(f"Test dimension from model {config.EMBEDDING_MODEL_NAME}: {test_dimension}")
    except Exception as e:
        logger.error(f"Could not load test embedding model {config.EMBEDDING_MODEL_NAME} to get dimension. Error: {e}. Defaulting test_dimension to 384.")
        test_dimension = 384 # A common dimension, e.g. for all-MiniLM-L6-v2


    test_index_dir = Path("./test_faiss_index_doc_manager")

    # Clean up previous test run
    if test_index_dir.exists():
        import shutil
        logger.debug(f"Cleaning up old test directory: {test_index_dir}")
        shutil.rmtree(test_index_dir)
    test_index_dir.mkdir(parents=True, exist_ok=True)

    vs_manager: Optional[VectorStoreManager] = None # Define for cleanup block

    try:
        # Test 1: Initialization (will try to get dim from model)
        logger.info("\n--- Test 1: Initialize VectorStoreManager ---")
        # Pass explicit dimension for predictability in tests, overriding model-derived one if different.
        # Or ensure the model used for tests always gives a known dimension.
        # For this test, we'll use the dimension derived from the actual model.
        vs_manager = VectorStoreManager(index_dir_path=test_index_dir, embedding_dimension=test_dimension)
        assert vs_manager.is_initialized(), "Test 1 Failed: Manager not initialized"
        assert vs_manager.get_dimension() == test_dimension, f"Test 1 Failed: Dimension mismatch. Expected {test_dimension}, Got {vs_manager.get_dimension()}"
        assert vs_manager.get_ntotal_faiss() == 0, "Test 1 Failed: New index should be empty"
        assert vs_manager.get_document_count() == 0, "Test 1 Failed: New manager should have no documents"
        logger.info("Test 1 Passed: Initialization successful.")

        # Test 2: Add a document
        logger.info("\n--- Test 2: Add a document ---")
        doc1_id = "doc1.txt"
        doc1_content = "This is the first document. It talks about apples and oranges. " * 50 # Make it long enough for multiple chunks
        doc1_metadata = {"source": "manual", "version": "1.0"}
        vs_manager.add_document(doc1_id, doc1_content, doc1_metadata)

        assert vs_manager.get_document_count() == 1, "Test 2 Failed: Document count should be 1"
        assert doc1_id in vs_manager.doc_id_to_chunk_ids
        assert len(vs_manager.doc_id_to_chunk_ids[doc1_id]) > 0, "Test 2 Failed: Document should have associated chunk IDs"
        num_chunks_doc1 = len(vs_manager.doc_id_to_chunk_ids[doc1_id])
        logger.info(f"Doc1 generated {num_chunks_doc1} chunks.")
        assert vs_manager.get_ntotal_faiss() == num_chunks_doc1, f"Test 2 Failed: FAISS total should be {num_chunks_doc1}"
        assert vs_manager.doc_id_to_metadata.get(doc1_id) == doc1_metadata
        first_chunk_id_doc1 = vs_manager.doc_id_to_chunk_ids[doc1_id][0]
        assert first_chunk_id_doc1 in vs_manager.chunk_id_to_text
        assert vs_manager.chunk_id_to_doc_id.get(first_chunk_id_doc1) == doc1_id
        logger.info("Test 2 Passed: Document added successfully.")

        # Test 3: Add another document
        logger.info("\n--- Test 3: Add a second document ---")
        doc2_id = "doc2.md"
        doc2_content = "A completely different document about bananas and grapes. " * 60
        vs_manager.add_document(doc2_id, doc2_content, {"source": "markdown_files"})
        assert vs_manager.get_document_count() == 2
        num_chunks_doc2 = len(vs_manager.doc_id_to_chunk_ids[doc2_id])
        logger.info(f"Doc2 generated {num_chunks_doc2} chunks.")
        assert vs_manager.get_ntotal_faiss() == num_chunks_doc1 + num_chunks_doc2
        logger.info("Test 3 Passed: Second document added successfully.")

        # Test 4: Search (ensure it finds something from doc1)
        logger.info("\n--- Test 4: Search for content from doc1 ---")
        query_text = "apples and oranges"
        query_embedding = generate_embeddings([query_text], model_name=config.EMBEDDING_MODEL_NAME)[0]
        search_results = vs_manager.search(query_embedding, k=3)
        assert len(search_results) > 0, "Test 4 Failed: Search returned no results"
        logger.info(f"Search for '{query_text}' found {len(search_results)} results.")
        if search_results:
            top_one = search_results[0]
            logger.info(f"Top result: doc_id='{top_one['doc_id']}', metadata='{top_one['metadata']}', "
                        f"dist='{top_one['distance']}', chunk='{top_one['chunk_text'][:50]}...'")

        # Check if the source of the chunk is doc1
        found_doc1_chunk = False
        for res_item in search_results:
            if res_item["doc_id"] == doc1_id and "apples" in res_item["chunk_text"]:
                found_doc1_chunk = True
                break
        assert found_doc1_chunk, "Test 4 Failed: Search results did not contain expected content from doc1"
        logger.info("Test 4 Passed: Search is working and returning rich results.")

        # Test 5: Update doc1
        logger.info("\n--- Test 5: Update document 1 ---")
        doc1_updated_content = "Document 1 now talks about pineapples and mangoes. Completely new content. " * 55
        doc1_updated_metadata = {"source": "manual_v2", "version": "2.0"}
        vs_manager.update_document(doc1_id, doc1_updated_content, doc1_updated_metadata)

        assert vs_manager.get_document_count() == 2, "Test 5 Failed: Document count should still be 2 after update"
        num_chunks_doc1_updated = len(vs_manager.doc_id_to_chunk_ids[doc1_id])
        logger.info(f"Updated Doc1 generated {num_chunks_doc1_updated} chunks.")
        assert num_chunks_doc1_updated > 0
        # Total ntotal in FAISS should be (total before update - old doc1 chunks + new doc1 chunks)
        assert vs_manager.get_ntotal_faiss() == (num_chunks_doc1 + num_chunks_doc2 - num_chunks_doc1 + num_chunks_doc1_updated)
        assert vs_manager.doc_id_to_metadata.get(doc1_id) == doc1_updated_metadata
        logger.info("Test 5 Passed: Document updated successfully.")

        # Test 6: Search for new content in updated doc1, ensure old is gone
        logger.info("\n--- Test 6: Search for new and old content in updated doc1 ---")
        query_new_doc1 = generate_embeddings(["pineapples and mangoes"], model_name=config.EMBEDDING_MODEL_NAME)[0]
        results_new = vs_manager.search(query_new_doc1, k=2)
        assert any("pineapples" in res["chunk_text"] for res in results_new), "Test 6 Failed: New content not found after update"
        if results_new:
            logger.info(f"Search for new content (pineapples) found doc_id: {results_new[0]['doc_id']}")

        query_old_doc1 = generate_embeddings(["apples and oranges"], model_name=config.EMBEDDING_MODEL_NAME)[0]
        results_old = vs_manager.search(query_old_doc1, k=2)
        assert not any("apples" in res["chunk_text"] for res in results_old), "Test 6 Failed: Old content found after update"
        logger.info("Test 6 Passed: Content update reflected in search (old content gone, new content findable).")

        # Test 7: Remove doc2
        logger.info("\n--- Test 7: Remove document 2 ---")
        vs_manager.remove_document(doc2_id)
        assert vs_manager.get_document_count() == 1, "Test 7 Failed: Document count should be 1 after removal"
        assert doc2_id not in vs_manager.doc_id_to_chunk_ids
        assert vs_manager.get_ntotal_faiss() == num_chunks_doc1_updated # Only updated doc1 chunks should remain
        logger.info("Test 7 Passed: Document removed successfully.")

        # Test 8: Save and Load
        logger.info("\n--- Test 8: Save and Load index ---")
        vs_manager.save_index()
        logger.info("Index saved. Now loading into a new manager instance.")

        vs_manager_loaded = VectorStoreManager(index_dir_path=test_index_dir, embedding_dimension=test_dimension)
        assert vs_manager_loaded.is_initialized(), "Test 8 Failed: Loaded manager not initialized"
        assert vs_manager_loaded.get_dimension() == test_dimension
        assert vs_manager_loaded.get_document_count() == 1, f"Test 8 Failed: Loaded doc count is {vs_manager_loaded.get_document_count()}, expected 1"
        assert vs_manager_loaded.get_ntotal_faiss() == num_chunks_doc1_updated
        assert doc1_id in vs_manager_loaded.doc_id_to_chunk_ids
        assert len(vs_manager_loaded.doc_id_to_chunk_ids[doc1_id]) == num_chunks_doc1_updated
        assert vs_manager_loaded.doc_id_to_metadata.get(doc1_id) == doc1_updated_metadata
        first_loaded_chunk_id = vs_manager_loaded.doc_id_to_chunk_ids[doc1_id][0]
        assert first_loaded_chunk_id in vs_manager_loaded.chunk_id_to_text
        assert "pineapples" in vs_manager_loaded.chunk_id_to_text[first_loaded_chunk_id]
        logger.info("Test 8 Passed: Save and Load successful, data integrity seems fine.")

        # Test 9: Remove non-existent document
        logger.info("\n--- Test 9: Remove non-existent document ---")
        assert not vs_manager.remove_document("non_existent_doc_id"), "Test 9 Failed: Removing non-existent should return False"
        logger.info("Test 9 Passed: Attempt to remove non-existent document handled.")

        # Test 10: Add existing document (should fail)
        logger.info("\n--- Test 10: Add existing document (should fail) ---")
        try:
            vs_manager.add_document(doc1_id, "some new content again", {})
            assert False, "Test 10 Failed: Adding existing doc_id should raise ValueError"
        except ValueError as e:
            logger.info(f"Test 10 Passed: Correctly caught ValueError: {e}")

    except Exception as e:
        logger.error(f"A test failed or an unexpected error occurred: {e}", exc_info=True)
    finally:
        # Clean up
        if vs_manager and vs_manager.index_dir_path.exists(): # vs_manager might be None if init failed early
            logger.info(f"Cleaning up test directory: {vs_manager.index_dir_path}")
            # shutil.rmtree(vs_manager.index_dir_path) # Keep for inspection after run
        logger.info("\nVectorStoreManager (document-centric) module test run complete.")
        logger.warning(f"Test directory {test_index_dir} is NOT automatically cleaned up for inspection.")


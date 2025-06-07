import faiss
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

# Get a logger for this module.
# Assumes that the application using this module will configure the root logger.
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages a FAISS vector index for storing and searching text embeddings.
    It uses IndexFlatL2 for L2 distance search and IndexIDMap to map vectors to original text chunk IDs.
    """
    def __init__(self, index_dir_path: Union[str, Path], dimension: Optional[int] = None) -> None:
        """
        Initializes the VectorStoreManager.

        Args:
            index_dir_path: Path to the directory where the FAISS index and metadata will be stored.
            dimension: The dimensionality of the embeddings. Required if creating a new index
                       and one doesn't already exist at index_dir_path.
        """
        self.index_dir_path = Path(index_dir_path)
        self.index_file = self.index_dir_path / "vector_store.faiss"
        self.metadata_file = self.index_dir_path / "vector_store_meta.pkl"

        self.index: Optional[faiss.IndexIDMap] = None
        self.chunk_map: List[str] = [] # Stores the original text chunks
        self.current_dimension: Optional[int] = None # Dimension of the loaded/created index

        self.load_index(required_dimension=dimension)

    def _create_new_index(self, dimension: int) -> None:
        """
        Creates a new FAISS index (IndexFlatL2 wrapped in IndexIDMap).
        """
        logger.info(f"Creating a new FAISS index with dimension {dimension} at {self.index_dir_path}")
        try:
            # IndexFlatL2 is for Euclidean distance.
            # For cosine similarity on normalized vectors, L2 distance works well.
            # If vectors are not normalized, use IndexFlatIP and normalize before adding/searching.
            base_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap(base_index)
            self.chunk_map = []
            self.current_dimension = dimension
            logger.info(f"Successfully created new FAISS index with dimension {dimension}.")
        except Exception as e:
            logger.error(f"Failed to create new FAISS index: {e}", exc_info=True)
            self.index = None
            self.current_dimension = None
            raise

    def add_embeddings(self, text_chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Adds text chunks and their corresponding embeddings to the FAISS index.

        Args:
            text_chunks: A list of original text strings.
            embeddings: A list of embeddings (each a list of floats) corresponding to text_chunks.

        Raises:
            ValueError: If the index is not initialized or if input lists are empty/mismatched.
        """
        if self.index is None:
            logger.error("FAISS index is not initialized. Cannot add embeddings.")
            raise ValueError("FAISS index not initialized. Call load_index() or ensure dimension is provided at init.")

        if not text_chunks or not embeddings:
            logger.warning("Text chunks or embeddings list is empty. Nothing to add.")
            return

        if len(text_chunks) != len(embeddings):
            logger.error(f"Mismatch between number of text chunks ({len(text_chunks)}) and embeddings ({len(embeddings)}).")
            raise ValueError("Number of text chunks and embeddings must be the same.")

        embeddings_np = np.array(embeddings, dtype='float32')

        # Optional: Normalize embeddings if your chosen model doesn't output normalized ones
        # and you're using IndexFlatIP or want L2 on normalized for cosine-like behavior.
        # faiss.normalize_L2(embeddings_np) # Uncomment if normalization is needed

        if embeddings_np.shape[1] != self.current_dimension:
            logger.error(f"Dimension of provided embeddings ({embeddings_np.shape[1]}) does not match index dimension ({self.current_dimension}).")
            raise ValueError("Dimension of embeddings does not match index dimension.")

        start_id = len(self.chunk_map)
        ids = np.arange(start_id, start_id + len(text_chunks))

        try:
            self.index.add_with_ids(embeddings_np, ids)
            self.chunk_map.extend(text_chunks)
            logger.info(f"Added {len(text_chunks)} new embeddings to the index. Total entries: {self.index.ntotal}.")
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}", exc_info=True)
            # Potentially revert additions to chunk_map if partial add is an issue, though add_with_ids is usually atomic.
            raise

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the FAISS index for the k nearest neighbors to the query_embedding.

        Args:
            query_embedding: A single embedding (list of floats) for the query.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains (text_chunk, distance).
            Distance is L2 distance for IndexFlatL2.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is not initialized or is empty. Cannot perform search.")
            return []

        if not query_embedding:
            logger.warning("Query embedding is empty. Cannot perform search.")
            return []

        query_embedding_np = np.array([query_embedding], dtype='float32')
        # Optional: Normalize query_embedding_np if embeddings in index were normalized and using IndexFlatIP
        # faiss.normalize_L2(query_embedding_np)

        if query_embedding_np.shape[1] != self.current_dimension:
            logger.error(f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match index dimension ({self.current_dimension}).")
            return []

        try:
            distances, ids = self.index.search(query_embedding_np, k)
            results: List[Tuple[str, float]] = []
            for i, doc_id in enumerate(ids[0]): # ids[0] because query is a single vector
                if doc_id != -1: # FAISS returns -1 for IDs not found or if k > ntotal
                    if 0 <= doc_id < len(self.chunk_map):
                        results.append((self.chunk_map[doc_id], float(distances[0][i])))
                    else:
                        logger.warning(f"Invalid document ID {doc_id} found in search results. Max ID is {len(self.chunk_map)-1}.")
            logger.info(f"Search returned {len(results)} results for k={k}.")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

    def save_index(self) -> None:
        """
        Saves the FAISS index and associated metadata (chunk_map, dimension) to disk.
        """
        if self.index is None:
            logger.warning("No index to save.")
            return

        try:
            self.index_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving FAISS index to {self.index_file}")
            faiss.write_index(self.index, str(self.index_file))

            metadata = {
                "chunk_map": self.chunk_map,
                "dimension": self.current_dimension
            }
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)
            logger.info(f"Successfully saved FAISS index and metadata to {self.index_dir_path}.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index or metadata: {e}", exc_info=True)
            raise

    def load_index(self, required_dimension: Optional[int] = None) -> None:
        """
        Loads the FAISS index and metadata from disk. If not found and required_dimension is provided,
        a new index is created.

        Args:
            required_dimension: The dimension the index should have. If an existing index is loaded,
                                its dimension is checked against this. If a new index is created,
                                this dimension is used.
        """
        if self.index_file.exists() and self.metadata_file.exists():
            logger.info(f"Attempting to load existing FAISS index from {self.index_dir_path}...")
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "rb") as f:
                    metadata = pickle.load(f)
                self.chunk_map = metadata.get("chunk_map", [])
                loaded_dimension = metadata.get("dimension")

                if loaded_dimension is None:
                    logger.error("Failed to load dimension from metadata. Index may be corrupt.")
                    self.index = None # Invalidate loaded index
                    # Fall through to potentially create new if required_dimension is set
                else:
                    self.current_dimension = loaded_dimension
                    logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} entries and dimension {self.current_dimension}.")

                if required_dimension is not None and self.current_dimension != required_dimension:
                    logger.critical(
                        f"Loaded index dimension ({self.current_dimension}) does not match "
                        f"required dimension ({required_dimension}). This can lead to errors."
                        "Consider deleting the old index or using the correct dimension."
                    )
                    # Decide on behavior: raise error, re-initialize, or just warn.
                    # For now, we'll allow it to proceed but with a critical warning.
                    # raise ValueError("Index dimension mismatch.")

            except Exception as e:
                logger.error(f"Failed to load FAISS index or metadata from {self.index_dir_path}: {e}", exc_info=True)
                self.index = None # Ensure index is None if loading fails
                self.chunk_map = []
                self.current_dimension = None
                # Fall through to create new if required_dimension is set

        # If index is still None (either not found, or loading failed)
        if self.index is None:
            if required_dimension is not None:
                logger.info(f"No existing index found or loading failed. Creating new index with dimension {required_dimension}.")
                try:
                    self._create_new_index(required_dimension)
                except Exception: # Catch creation failure
                    logger.error(f"Failed to create a new index with dimension {required_dimension} after previous load attempt failed or index was missing.")
                    # self.index remains None, self.current_dimension remains None
            else:
                logger.warning(
                    "No FAISS index found at specified path and no dimension provided to create a new one. "
                    "Index remains uninitialized."
                )

    def get_dimension(self) -> Optional[int]:
        """Returns the dimension of the loaded/created FAISS index."""
        if self.index:
            return self.index.d
        return self.current_dimension # Might be set from metadata even if index object itself failed to load temporarily

    def get_ntotal(self) -> int:
        """Returns the total number of embeddings in the FAISS index."""
        if self.index:
            return self.index.ntotal
        return 0

    def is_initialized(self) -> bool:
        """Checks if the FAISS index is initialized and ready."""
        return self.index is not None and self.current_dimension is not None


if __name__ == '__main__':
    # For standalone testing of this module.
    logging.basicConfig(
        level=logging.DEBUG, # More verbose for direct testing
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Running VectorStoreManager module directly for testing with DEBUG level logging.")

    test_index_dir = Path("./test_faiss_index_data")
    test_dimension = 3 # Small dimension for testing

    # Clean up previous test run
    if test_index_dir.exists():
        import shutil
        logger.debug(f"Cleaning up old test directory: {test_index_dir}")
        shutil.rmtree(test_index_dir)

    # Test 1: Initialization and creation of a new index
    logger.info("\n--- Test 1: Initialize and create new index ---")
    try:
        vs_manager = VectorStoreManager(index_dir_path=test_index_dir, dimension=test_dimension)
        assert vs_manager.is_initialized(), "Test 1 Failed: Manager not initialized"
        assert vs_manager.get_dimension() == test_dimension, f"Test 1 Failed: Dimension mismatch. Expected {test_dimension}, Got {vs_manager.get_dimension()}"
        assert vs_manager.get_ntotal() == 0, "Test 1 Failed: New index should be empty"
        logger.info("Test 1 Passed: Initialization and creation successful.")
    except Exception as e:
        logger.error(f"Test 1 Failed: {e}", exc_info=True)

    # Test 2: Adding embeddings
    logger.info("\n--- Test 2: Add embeddings ---")
    if 'vs_manager' in locals() and vs_manager.is_initialized():
        try:
            sample_chunks = ["chunk1", "chunk2", "chunk3"]
            # Embeddings must match test_dimension (3)
            sample_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            vs_manager.add_embeddings(sample_chunks, sample_embeddings)
            assert vs_manager.get_ntotal() == 3, "Test 2 Failed: Incorrect number of embeddings"
            logger.info("Test 2 Passed: Embeddings added successfully.")

            # Test adding mismatched dimension
            try:
                vs_manager.add_embeddings(["wrong_dim_chunk"], [[0.1, 0.2]]) # Dim 2
                logger.error("Test 2 Failed: Should have raised ValueError for dimension mismatch.")
            except ValueError:
                logger.info("Test 2 Sub-test Passed: Correctly caught dimension mismatch on add.")

        except Exception as e:
            logger.error(f"Test 2 Failed: {e}", exc_info=True)
    else:
        logger.warning("Skipping Test 2 as VectorStoreManager not initialized from Test 1.")

    # Test 3: Searching embeddings
    logger.info("\n--- Test 3: Search embeddings ---")
    if 'vs_manager' in locals() and vs_manager.is_initialized() and vs_manager.get_ntotal() > 0:
        try:
            query_emb = [0.15, 0.25, 0.35] # Close to [0.1, 0.2, 0.3]
            results = vs_manager.search(query_emb, k=1)
            assert len(results) == 1, "Test 3 Failed: Incorrect number of search results"
            assert results[0][0] == "chunk1", f"Test 3 Failed: Unexpected search result. Expected 'chunk1', Got '{results[0][0]}'"
            logger.info(f"Test 3 Passed: Search successful. Closest: {results[0]}")
        except Exception as e:
            logger.error(f"Test 3 Failed: {e}", exc_info=True)
    else:
        logger.warning("Skipping Test 3 as prerequisites not met.")

    # Test 4: Saving the index
    logger.info("\n--- Test 4: Save index ---")
    if 'vs_manager' in locals() and vs_manager.is_initialized():
        try:
            vs_manager.save_index()
            assert (test_index_dir / "vector_store.faiss").exists(), "Test 4 Failed: FAISS index file not saved"
            assert (test_index_dir / "vector_store_meta.pkl").exists(), "Test 4 Failed: Metadata file not saved"
            logger.info("Test 4 Passed: Index saved successfully.")
        except Exception as e:
            logger.error(f"Test 4 Failed: {e}", exc_info=True)
    else:
        logger.warning("Skipping Test 4 as VectorStoreManager not initialized.")

    # Test 5: Loading the index
    logger.info("\n--- Test 5: Load index ---")
    if 'vs_manager' in locals(): # vs_manager might exist even if initialization failed, so check files
        if (test_index_dir / "vector_store.faiss").exists():
            try:
                del vs_manager # Delete previous instance
                vs_manager_loaded = VectorStoreManager(index_dir_path=test_index_dir, dimension=test_dimension) # Provide dimension for safety
                assert vs_manager_loaded.is_initialized(), "Test 5 Failed: Manager not initialized after load"
                assert vs_manager_loaded.get_dimension() == test_dimension, f"Test 5 Failed: Dimension mismatch after load. Expected {test_dimension}, Got {vs_manager_loaded.get_dimension()}"
                assert vs_manager_loaded.get_ntotal() == 3, f"Test 5 Failed: Incorrect Ntotal after load. Expected 3, Got {vs_manager_loaded.get_ntotal()}"
                logger.info(f"Test 5 Passed: Index loaded successfully with {vs_manager_loaded.get_ntotal()} entries.")

                # Search again with loaded index
                query_emb = [0.75, 0.85, 0.95] # Close to [0.7, 0.8, 0.9]
                results_loaded = vs_manager_loaded.search(query_emb, k=1)
                assert len(results_loaded) == 1, "Test 5 Failed: Search after load returned incorrect number of results"
                assert results_loaded[0][0] == "chunk3", f"Test 5 Failed: Unexpected search result after load. Expected 'chunk3', Got '{results_loaded[0][0]}'"
                logger.info(f"Test 5 Passed: Search after load successful. Closest: {results_loaded[0]}")

            except Exception as e:
                logger.error(f"Test 5 Failed: {e}", exc_info=True)
        else:
            logger.warning("Skipping Test 5 as index files from Test 4 do not exist.")
    else:
         logger.warning("Skipping Test 5 as VectorStoreManager not initialized in previous tests.")

    # Test 6: Dimension mismatch when loading
    logger.info("\n--- Test 6: Load index with dimension mismatch ---")
    if (test_index_dir / "vector_store.faiss").exists():
        try:
            # vs_manager_loaded should have dimension test_dimension (e.g. 3)
            # Try to load it but require a different dimension
            wrong_dimension = test_dimension + 1
            logger.info(f"Attempting to load index (dim={test_dimension}) with required_dimension={wrong_dimension}")
            # Suppress critical log for this specific test to avoid clutter if it's expected
            logging.getLogger('ai_tech_support_agent.app.logic.vector_store_manager').setLevel(logging.ERROR)
            vs_manager_mismatch = VectorStoreManager(index_dir_path=test_index_dir, dimension=wrong_dimension)
            # Reset log level
            logging.getLogger('ai_tech_support_agent.app.logic.vector_store_manager').setLevel(logging.INFO)

            # Behavior here depends on how strictly we handle mismatch.
            # Current code logs critical error but proceeds.
            # For a strict test, you might expect a ValueError during __init__ or load_index.
            # Here, we check if the loaded dimension is the original one, despite the 'required' one.
            assert vs_manager_mismatch.get_dimension() == test_dimension, \
                f"Test 6 Failed: Loaded dimension should be {test_dimension}, not {vs_manager_mismatch.get_dimension()}"
            logger.info(f"Test 6 Passed: Dimension mismatch handled (logged critical error, loaded original dimension {vs_manager_mismatch.get_dimension()}).")

        except Exception as e: # If strict error handling (e.g. raise ValueError) was added
            logger.info(f"Test 6 Passed with expected error: {e}")
        except AssertionError as e:
             logger.error(f"Test 6 Failed: {e}", exc_info=True)
    else:
        logger.warning("Skipping Test 6 as index files do not exist.")


    logger.info("\nVectorStoreManager module test run complete.")
    # Optional: Clean up test directory after tests
    # if test_index_dir.exists():
    #     import shutil
    #     shutil.rmtree(test_index_dir)
    #     logger.debug(f"Cleaned up test directory: {test_index_dir}")

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field # Added Field

# Project-specific imports
from app import config
from app.logic.embedding_generator import EmbeddingModel, generate_embeddings
from app.logic.vector_store_manager import VectorStoreManager
from app.logic.llm_handler import LLMGenerator # Added LLMGenerator

# Configure logging
# The general logging configuration for the application.
# Uvicorn will have its own loggers (uvicorn, uvicorn.access, uvicorn.error)
# which can be configured separately if needed, often via Uvicorn's command-line options
# or programmatically if running Uvicorn with `uvicorn.run()`.
# This basicConfig will apply to loggers created by `logging.getLogger()` in our app modules.

logging.basicConfig(
    level=config.LOG_LEVEL,  # Set the root logger level
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # handlers=[logging.StreamHandler()] # Ensures logs go to stderr/stdout
)

# Get a logger for this specific module
logger = logging.getLogger(__name__)

# Example: If you want to specifically control uvicorn's access log level via python
# uvicorn_access_logger = logging.getLogger("uvicorn.access")
# uvicorn_access_logger.setLevel(logging.WARNING) # Or whatever level you prefer

app = FastAPI(
    title="AI Tech Support Agent API",
    description="API for querying a knowledge base using semantic search.",
    version="0.1.0"
)

# --- Global components ---
# Initialize and configure the embedding model
# Setting the model name here influences which model is loaded by EmbeddingModel.get_model()
EmbeddingModel.set_model_name(config.MODEL_NAME)

# Initialize the Vector Store Manager
# The dimension is crucial for creating the index if it doesn't exist on first run.
try:
    vector_store_manager = VectorStoreManager(
        index_dir_path=config.VECTOR_STORE_INDEX_DIR,
        dimension=config.EMBEDDING_DIMENSION
    )
except Exception as e:
    logger.error(f"Fatal error during VectorStoreManager initialization: {e}", exc_info=True)
    # If the vector store is critical, you might want to prevent app startup.
    # For now, we'll log and the app will likely fail on /query if it's not ready.
    vector_store_manager = None


# --- Pydantic Models for API requests/responses ---
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    text: str
    score: float = Field(default=0.0, description="Retrieval score from vector store (e.g., L2 distance)")
    # Optional: Add source_document: Optional[str] = None if this metadata becomes available

class QueryResponse(BaseModel):
    generated_answer: str
    source_chunks: List[SourceChunk]
    message: Optional[str] = None # For any additional info, like warnings


# --- FastAPI event handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing models and vector store...")
    # Initialize Embedding Model
    try:
        embedding_model_instance = EmbeddingModel.get_model() # Ensures embedding model is loaded
        logger.info(f"Embedding model '{EmbeddingModel._model_name}' (device: {EmbeddingModel._device}) loaded successfully.")
    except Exception as e:
        logger.error(f"Error during Embedding Model initialization: {e}", exc_info=True)
        # Depending on severity, might want to raise to stop app or set a health check status

    # Initialize Vector Store Manager
    # Note: vector_store_manager is already initialized globally when main.py is loaded.
    # This section in startup is more for logging its status or performing checks.
    if vector_store_manager:
        logger.info(f"Vector store manager path: {config.VECTOR_STORE_INDEX_DIR}")
        logger.info(f"Vector store contains {vector_store_manager.get_ntotal()} items.")

        store_dim = vector_store_manager.get_dimension()
        if store_dim is not None and store_dim != config.EMBEDDING_DIMENSION:
            logger.critical(
                f"CRITICAL Dimension Mismatch: Vector store dimension is {store_dim}, "
                f"but configured EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}. "
                "This will likely lead to errors during embedding generation or search. "
                "Please ensure your vector store is compatible with the configured model."
            )
        elif store_dim is None and vector_store_manager.get_ntotal() > 0:
                logger.warning("Vector store has items but its dimension could not be determined from metadata. This might indicate a corrupted metadata file.")
        elif store_dim is None and vector_store_manager.get_ntotal() == 0:
                logger.info("Vector store is empty and dimension is not yet set (will be set on first add_embeddings).")
    else:
        logger.error("VectorStoreManager is not available. Query functionality will be impaired if it was intended to be used.")

    # Initialize LLM Generator
    logger.info(f"Attempting to initialize LLM: {config.LLM_MODEL_NAME}...")
    try:
        LLMGenerator._initialize(config.LLM_MODEL_NAME) # Trigger LLM model loading
        if LLMGenerator._model and LLMGenerator._tokenizer:
            logger.info(f"LLM '{LLMGenerator._model_name_loaded}' (device: {LLMGenerator._device}, model actual device: {LLMGenerator._model.device}) initialized successfully.")
        else:
            logger.error(f"LLM '{config.LLM_MODEL_NAME}' failed to initialize properly (model or tokenizer is None).")
            # This state will cause /query to fail if LLM is required.
    except Exception as e:
        logger.error(f"Error during LLM initialization: {e}", exc_info=True)
        # App will continue to run, but /query endpoint will likely fail if LLM is needed.

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Accepts a user's question, generates an embedding for it,
    queries the vector store for relevant document chunks,
    then uses an LLM to generate an answer based on the context.
    """
    logger.info(f"Received query: '{request.question}', top_k={request.top_k if request.top_k is not None else 'default'}")

    # --- Check for LLM availability first ---
    if not LLMGenerator._model or not LLMGenerator._tokenizer:
        logger.error("LLM model or tokenizer is not available. Cannot process query requiring LLM.")
        raise HTTPException(status_code=503, detail="LLM service is not ready. Please try again later or contact support.")

    # --- Vector Store Operations ---
    if not vector_store_manager or not vector_store_manager.is_initialized():
        logger.error("Vector store is not available or not initialized.")
        raise HTTPException(status_code=503, detail="Vector store is not ready. Please try again later.")

    if vector_store_manager.get_ntotal() == 0:
        logger.warning("Query attempted on an empty vector store.")
        # Generate a response using LLM without context
        try:
            logger.info("Vector store is empty. Attempting to generate answer without specific context...")
            llm_answer_no_context = LLMGenerator.generate_answer(request.question, [])
            return QueryResponse(
                generated_answer=llm_answer_no_context,
                source_chunks=[],
                message="The knowledge base is currently empty. The answer is generated without specific context."
            )
        except Exception as e:
            logger.error(f"Error generating answer without context when store is empty: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate answer and knowledge base is empty.")

    # 1. Generate embedding for the query
    try:
        query_embeddings = generate_embeddings(
            text_chunks=[request.question],
            model_name=config.MODEL_NAME
        )
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to generate embedding for the query.")
            raise HTTPException(status_code=500, detail="Could not generate query embedding.")
        query_embedding = query_embeddings[0]
    except Exception as e:
        logger.error(f"Exception during query embedding generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating query embedding.")

    # 2. Determine k for search and retrieve chunks
    k = request.top_k if request.top_k is not None and request.top_k > 0 else config.TOP_K_RESULTS
    logger.debug(f"Searching vector store with k={k}.")
    try:
        retrieved_results: List[Tuple[str, float]] = vector_store_manager.search(query_embedding, k=k)
        if not retrieved_results:
            logger.info("No relevant chunks found in vector store for the query.")
            # Decide: try to answer with LLM without context, or state no relevant info found.
            # For now, let's try to answer with LLM without context.
            llm_answer_no_chunks = LLMGenerator.generate_answer(request.question, [])
            return QueryResponse(
                generated_answer=llm_answer_no_chunks,
                source_chunks=[],
                message="No specific information found in the knowledge base for this query. The answer is generated without specific context."
            )
    except Exception as e:
        logger.error(f"Exception during vector store search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error searching for documents in the vector store.")

    # 3. Prepare context for LLM
    context_texts = [chunk_text for chunk_text, score in retrieved_results]
    logger.info(f"Retrieved {len(context_texts)} chunks for LLM context. Total characters: {sum(len(t) for t in context_texts)}")

    # 4. Generate Answer using LLM
    try:
        logger.debug("Sending query and context to LLM for answer generation.")
        llm_answer = LLMGenerator.generate_answer(request.question, context_texts)
        if llm_answer.startswith("Error:"): # Check for errors from LLMGenerator itself
             logger.error(f"LLMGenerator returned an error: {llm_answer}")
             raise HTTPException(status_code=500, detail=f"LLM failed to generate an answer: {llm_answer}")
    except Exception as e: # Catch any other unexpected error from generate_answer
        logger.error(f"Exception during LLM answer generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating answer from LLM.")

    # 5. Construct and Return Response
    source_chunks_for_response = [
        SourceChunk(text=chunk_text, score=score) for chunk_text, score in retrieved_results
    ]

    logger.info(f"Successfully generated answer for query: '{request.question}'.")
    return QueryResponse(generated_answer=llm_answer, source_chunks=source_chunks_for_response)


@app.get("/demo_query/", response_model=QueryResponse)
async def demo_query():
    """
    Returns a predefined QueryResponse object for demonstration purposes.
    """
    logger.info("Accessed /demo_query/ endpoint.")

    sample_question = "How to reset my password?"
    sample_answer = (
        "To reset your password, please follow these steps:\n"
        "1. Go to the login page.\n"
        "2. Click on the 'Forgot Password?' link.\n"
        "3. Enter your email address and follow the instructions sent to your inbox."
    )
    sample_chunks = [
        SourceChunk(text="Step 1: Navigate to the main login screen of the application.", score=0.92),
        SourceChunk(text="Step 2: Locate and click the 'Forgot Password?' or 'Reset Password' link, usually found below the login fields.", score=0.88),
        SourceChunk(text="Additional Info: Ensure you have access to the email account associated with your user profile to receive the reset link.", score=0.85)
    ]

    return QueryResponse(
        generated_answer=sample_answer,
        source_chunks=sample_chunks,
        message="This is a demo response."
    )

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    llm_status = "Not initialized"
    if LLMGenerator._model and LLMGenerator._tokenizer:
        llm_status = f"Initialized: {LLMGenerator._model_name_loaded} on {LLMGenerator._device} (model device: {LLMGenerator._model.device})"
    elif LLMGenerator._model_name_loaded : # Attempted to load but failed
        llm_status = f"Initialization failed for {LLMGenerator._model_name_loaded}"


    return {
        "message": "Welcome to the AI Tech Support Agent API.",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "vector_store_status": {
            "initialized": vector_store_manager.is_initialized() if vector_store_manager else False,
            "item_count": vector_store_manager.get_ntotal() if vector_store_manager else 0,
            "dimension": vector_store_manager.get_dimension() if vector_store_manager else None,
        },
        "embedding_model_status": {
            "model_name": EmbeddingModel._model_name,
            "device": EmbeddingModel._device,
            "initialized": EmbeddingModel._model is not None
        },
        "llm_status": llm_status,
        "configured_embedding_model": config.MODEL_NAME,
        "configured_embedding_dimension": config.EMBEDDING_DIMENSION,
        "configured_llm_model": config.LLM_MODEL_NAME,
    }

if __name__ == "__main__":
            logger.info(f"Vector store contains {vector_store_manager.get_ntotal()} items.")

            store_dim = vector_store_manager.get_dimension()
            if store_dim is not None and store_dim != config.EMBEDDING_DIMENSION:
                logger.critical(
                    f"CRITICAL Dimension Mismatch: Vector store dimension is {store_dim}, "
                    f"but configured EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}. "
                    "This will likely lead to errors during embedding generation or search. "
                    "Please ensure your vector store is compatible with the configured model."
                )
            elif store_dim is None and vector_store_manager.get_ntotal() > 0:
                 logger.warning("Vector store has items but its dimension could not be determined from metadata. This might indicate a corrupted metadata file.")
            elif store_dim is None and vector_store_manager.get_ntotal() == 0:
                 logger.info("Vector store is empty and dimension is not yet set (will be set on first add_embeddings).")

        else:
            logger.error("VectorStoreManager is not available. Query functionality will be impaired.")

    except Exception as e:
        logger.error(f"Error during startup model or vector store initialization: {e}", exc_info=True)
        # Depending on severity, you might want to raise an error to stop FastAPI from starting,
        # or set a global flag indicating the app is not healthy.

# --- API Endpoints ---
@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Accepts a user's question, generates an embedding for it,
    and queries the vector store for the most relevant document chunks.
    """
    logger.info(f"Received query: '{request.question}', top_k={request.top_k}")

    if not vector_store_manager or not vector_store_manager.is_initialized():
        logger.error("Vector store is not available or not initialized.")
        raise HTTPException(status_code=503, detail="Vector store is not ready. Please try again later.")

    if vector_store_manager.get_ntotal() == 0:
        logger.warning("Query attempted on an empty vector store.")
        return QueryResponse(results=[], message="The knowledge base is currently empty. No documents to search.")

    # 1. Generate embedding for the query
    try:
        # generate_embeddings expects a list of texts and returns a list of embeddings
        query_embeddings = generate_embeddings(
            text_chunks=[request.question],
            model_name=config.MODEL_NAME # Use the model name from config
        )
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to generate embedding for the query.")
            raise HTTPException(status_code=500, detail="Could not generate query embedding.")
        query_embedding = query_embeddings[0]
    except Exception as e:
        logger.error(f"Exception during query embedding generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating query embedding.")

    # 2. Determine k for search
    k = request.top_k if request.top_k is not None and request.top_k > 0 else config.TOP_K_RESULTS

    # 3. Search the vector store
    try:
        logger.debug(f"Searching vector store with k={k}. Query embedding dims: {len(query_embedding)}")
        search_results: List[Tuple[str, float]] = vector_store_manager.search(query_embedding, k=k)

        response_items = [QueryResponseItem(text_chunk=chunk, score=score) for chunk, score in search_results]

        logger.info(f"Found {len(response_items)} relevant chunks for query '{request.question}'.")
        return QueryResponse(results=response_items)

    except Exception as e:
        logger.error(f"Exception during vector store search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error searching for documents in the vector store.")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {
        "message": "Welcome to the AI Tech Support Agent API.",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "vector_store_status": {
            "initialized": vector_store_manager.is_initialized() if vector_store_manager else False,
            "item_count": vector_store_manager.get_ntotal() if vector_store_manager else 0,
            "dimension": vector_store_manager.get_dimension() if vector_store_manager else None,
            "configured_model": config.MODEL_NAME,
            "configured_dimension": config.EMBEDDING_DIMENSION
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server for local development on {config.API_HOST}:{config.API_PORT}")
    # Note: For production, use a proper ASGI server like Gunicorn with Uvicorn workers.
    # Example: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind {config.API_HOST}:{config.API_PORT}
    # Uvicorn's log_level can be set here. It will respect this for its own logs.
    # Our app's logging (basicConfig above) is separate but will also output.
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower() # Uvicorn's own log level setting
    )

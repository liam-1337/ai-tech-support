import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import uuid

# Project-specific imports
from app import config
from app.logic.embedding_generator import EmbeddingModel, generate_embeddings
from app.logic.vector_store_manager import VectorStoreManager
from app.logic.llm_handler import LLMGenerator
from app.session_manager import session_manager
from app.feedback_logger import setup_feedback_logger, log_interaction
from app.escalation_manager import check_for_escalation, create_escalation_ticket # Import escalation functions

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Tech Support Agent API",
    description="API for querying a knowledge base using semantic search.",
    version="0.1.0"
)

# --- Global components ---
EmbeddingModel.set_model_name(config.MODEL_NAME)
try:
    vector_store_manager = VectorStoreManager(
        index_dir_path=config.VECTOR_STORE_INDEX_DIR,
        dimension=config.EMBEDDING_DIMENSION
    )
except Exception as e:
    logger.error(f"Fatal error during VectorStoreManager initialization: {e}", exc_info=True)
    vector_store_manager = None

# --- Pydantic Models for API requests/responses ---
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    session_id: Optional[str] = Field(default=None, description="The ID of the current conversation session. If None, a new session will be created.")

class SourceChunk(BaseModel):
    chunk_text: str = Field(description="The text content of the retrieved chunk.")
    doc_id: str = Field(description="The identifier of the source document.")
    metadata: Dict = Field(default_factory=dict, description="Metadata associated with the source document.")
    distance: float = Field(default=0.0, description="Retrieval score/distance from vector store (e.g., L2 distance).")

class QueryResponse(BaseModel):
    generated_answer: str
    source_chunks: List[SourceChunk]
    session_id: str = Field(description="Identifier for the conversation session. Use this ID in subsequent requests to maintain context.")
    message: Optional[str] = None

# --- FastAPI event handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing models and vector store...")
    try:
        EmbeddingModel.get_model()
        logger.info(f"Embedding model '{EmbeddingModel._model_name}' (device: {EmbeddingModel._device}) loaded successfully.")
    except Exception as e:
        logger.error(f"Error during Embedding Model initialization: {e}", exc_info=True)

    if vector_store_manager:
        logger.info(f"Vector store manager path: {config.VECTOR_STORE_INDEX_DIR}")
        logger.info(f"Vector store contains {vector_store_manager.get_ntotal_faiss()} items.")
        store_dim = vector_store_manager.get_dimension()
        if store_dim is not None and store_dim != config.EMBEDDING_DIMENSION:
            logger.critical(
                f"CRITICAL Dimension Mismatch: Vector store dimension is {store_dim}, "
                f"but configured EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}."
            )
        elif store_dim is None and vector_store_manager.get_ntotal_faiss() > 0:
                logger.warning("Vector store has items but its dimension could not be determined from metadata.")
        elif store_dim is None and vector_store_manager.get_ntotal_faiss() == 0:
            logger.info("Vector store is empty and dimension is not yet set.")
    else:
        logger.error("VectorStoreManager is not available.")

    logger.info(f"Attempting to initialize LLM: {config.LLM_MODEL_NAME}...")
    try:
        LLMGenerator._initialize(config.LLM_MODEL_NAME)
        if LLMGenerator._model and LLMGenerator._tokenizer:
            logger.info(f"LLM '{LLMGenerator._model_name_loaded}' (device: {LLMGenerator._device}, model actual device: {LLMGenerator._model.device}) initialized successfully.")
        else:
            logger.error(f"LLM '{config.LLM_MODEL_NAME}' failed to initialize properly.")
    except Exception as e:
        logger.error(f"Error during LLM initialization: {e}", exc_info=True)

    try:
        setup_feedback_logger()
        logger.info("Feedback logger setup attempted during startup.")
    except Exception as e:
        logger.error(f"Error setting up feedback logger during startup: {e}", exc_info=True)

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    logger.info(f"Received query: '{request.question}', top_k={request.top_k if request.top_k is not None else 'default'}, session_id='{request.session_id}'")

    if not LLMGenerator._model or not LLMGenerator._tokenizer:
        logger.error("LLM model or tokenizer is not available.")
        raise HTTPException(status_code=503, detail="LLM service is not ready.")

    current_session_id: str
    if request.session_id and session_manager.session_exists(request.session_id):
        current_session_id = request.session_id
        logger.info(f"Continuing session: {current_session_id}")
    else:
        current_session_id = session_manager.create_session()
        logger.info(f"Created new session: {current_session_id}")
        if request.session_id:
            logger.warning(f"Client sent session_id '{request.session_id}' but it was not found. New session '{current_session_id}' created.")

    conversation_history = session_manager.get_session_history(current_session_id)
    logger.debug(f"Retrieved {len(conversation_history)} history items for session {current_session_id}.")

    # --- Escalation Check ---
    should_escalate, escalation_reason = check_for_escalation(
        current_session_id, request.question, conversation_history
    )
    if should_escalate:
        ticket_id = create_escalation_ticket(
            current_session_id, request.question, conversation_history, escalation_reason
        )
        escalation_message = (
            f"I understand this issue requires further attention. I've created a ticket for you: {ticket_id}. "
            "A human support agent will get back to you shortly."
        )
        session_manager.add_turn_to_history(current_session_id, request.question, escalation_message)
        log_interaction(
            session_id=current_session_id,
            user_query=request.question,
            retrieved_context_summary=[], # No context retrieval attempted for escalation
            llm_response=escalation_message,
            # Potentially add a field like "interaction_type": "escalation" to log_interaction if needed
        )
        return QueryResponse(
            generated_answer=escalation_message,
            source_chunks=[],
            session_id=current_session_id,
            message="This issue has been escalated."
        )

    retrieved_search_results: List[Dict[str, Any]] = []

    if not vector_store_manager or not vector_store_manager.is_initialized():
        logger.error("Vector store is not available or not initialized.")
        llm_answer_no_vs = LLMGenerator.generate_answer(
            query=request.question, context_chunks=[], conversation_history=conversation_history
        )
        session_manager.add_turn_to_history(current_session_id, request.question, llm_answer_no_vs)
        log_interaction(
            session_id=current_session_id, user_query=request.question,
            retrieved_context_summary=[], llm_response=llm_answer_no_vs
        )
        return QueryResponse(
            generated_answer=llm_answer_no_vs, source_chunks=[], session_id=current_session_id,
            message="Vector store is currently unavailable. Answer based on general knowledge and history."
        )

    if vector_store_manager.get_ntotal_faiss() == 0:
        logger.warning("Query attempted on an empty vector store.")
        try:
            llm_answer_no_context = LLMGenerator.generate_answer(
                query=request.question, context_chunks=[], conversation_history=conversation_history
            )
            session_manager.add_turn_to_history(current_session_id, request.question, llm_answer_no_context)
            log_interaction(
                session_id=current_session_id, user_query=request.question,
                retrieved_context_summary=[], llm_response=llm_answer_no_context
            )
            return QueryResponse(
                generated_answer=llm_answer_no_context, source_chunks=[], session_id=current_session_id,
                message="Knowledge base is empty. Answer based on general knowledge and history."
            )
        except Exception as e:
            logger.error(f"Error generating answer (no context) when store is empty: {e}", exc_info=True)
            log_interaction(
                session_id=current_session_id, user_query=request.question,
                retrieved_context_summary=[],
                llm_response=f"Error: Failed to generate answer (empty KB). Details: {str(e)}"
            )
            raise HTTPException(status_code=500, detail="Failed to generate answer (empty KB).")

    try:
        query_embeddings = generate_embeddings([request.question], model_name=config.MODEL_NAME)
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to generate embedding for the query.")
            raise HTTPException(status_code=500, detail="Could not generate query embedding.")
        query_embedding = query_embeddings[0]
    except Exception as e:
        logger.error(f"Exception during query embedding generation: {e}", exc_info=True)
        log_interaction(
            session_id=current_session_id, user_query=request.question,
            retrieved_context_summary=[],
            llm_response=f"Error: Failed to generate query embedding. Details: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Error generating query embedding.")

    k = request.top_k if request.top_k is not None and request.top_k > 0 else config.TOP_K_RESULTS
    logger.debug(f"Searching vector store with k={k}.")
    try:
        retrieved_search_results = vector_store_manager.search(query_embedding, k=k)
        if not retrieved_search_results:
            logger.info("No relevant chunks found in vector store for the query.")
            llm_answer_no_chunks = LLMGenerator.generate_answer(
                query=request.question, context_chunks=[], conversation_history=conversation_history
            )
            session_manager.add_turn_to_history(current_session_id, request.question, llm_answer_no_chunks)
            log_interaction(
                session_id=current_session_id, user_query=request.question,
                retrieved_context_summary=[], llm_response=llm_answer_no_chunks
            )
            return QueryResponse(
                generated_answer=llm_answer_no_chunks, source_chunks=[], session_id=current_session_id,
                message="No specific information found. Answer based on general knowledge and history."
            )
    except Exception as e:
        logger.error(f"Exception during vector store search: {e}", exc_info=True)
        log_interaction(
            session_id=current_session_id, user_query=request.question,
            retrieved_context_summary=[],
            llm_response=f"Error: Exception during vector store search. Details: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Error searching vector store.")

    context_texts = [result_item["chunk_text"] for result_item in retrieved_search_results]
    logger.info(f"Retrieved {len(context_texts)} chunks for LLM context.")

    try:
        logger.debug("Sending query, context, and history to LLM for answer generation.")
        llm_answer = LLMGenerator.generate_answer(
            query=request.question, context_chunks=context_texts, conversation_history=conversation_history
        )
        if llm_answer.startswith("Error:"):
             logger.error(f"LLMGenerator returned an error: {llm_answer}")
             session_manager.add_turn_to_history(current_session_id, request.question, llm_answer)
             log_interaction(
                session_id=current_session_id, user_query=request.question,
                retrieved_context_summary=[{"doc_id":item.get("doc_id"), "distance":item.get("distance")} for item in retrieved_search_results],
                llm_response=llm_answer
             )
             raise HTTPException(status_code=500, detail=f"LLM failed to generate an answer: {llm_answer}")
    except Exception as e:
        logger.error(f"Exception during LLM answer generation: {e}", exc_info=True)
        log_interaction(
            session_id=current_session_id, user_query=request.question,
            retrieved_context_summary=[{"doc_id":item.get("doc_id"), "distance":item.get("distance")} for item in retrieved_search_results],
            llm_response=f"Error: Exception during LLM answer generation. Details: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Error generating answer from LLM.")

    session_manager.add_turn_to_history(current_session_id, request.question, llm_answer)

    context_summary_for_log = [
        {
            "doc_id": item.get("doc_id"), "distance": item.get("distance"),
            "chunk_preview": item.get("chunk_text", "")[:100] + "..." if item.get("chunk_text") else None,
            "doc_metadata_source": item.get("metadata", {}).get("source")
        } for item in retrieved_search_results
    ]
    log_interaction(
        session_id=current_session_id, user_query=request.question,
        retrieved_context_summary=context_summary_for_log, llm_response=llm_answer
    )

    source_chunks_for_response = [
        SourceChunk(
            chunk_text=result_item["chunk_text"], doc_id=result_item["doc_id"],
            metadata=result_item["metadata"], distance=result_item["distance"]
        ) for result_item in retrieved_search_results
    ]

    logger.info(f"Successfully generated answer for query: '{request.question}'.")
    return QueryResponse(
        generated_answer=llm_answer, source_chunks=source_chunks_for_response, session_id=current_session_id
    )

@app.get("/demo_query/", response_model=QueryResponse)
async def demo_query():
    logger.info("Accessed /demo_query/ endpoint.")
    sample_question = "How to reset my password?"
    sample_answer = (
        "To reset your password, please follow these steps:\n"
        "1. Go to the login page.\n"
        "2. Click on the 'Forgot Password?' link.\n"
        "3. Enter your email address and follow the instructions sent to your inbox."
    )
    sample_chunks = [
        SourceChunk(
            chunk_text="Step 1: Navigate to the main login screen of the application.",
            doc_id="KB001.md", metadata={"title": "Password Reset Guide", "category": "Accounts"}, distance=0.92
        ),
        SourceChunk(
            chunk_text="Step 2: Locate and click the 'Forgot Password?' or 'Reset Password' link, usually found below the login fields.",
            doc_id="KB001.md", metadata={"title": "Password Reset Guide", "category": "Accounts"}, distance=0.88
        ),
        SourceChunk(
            chunk_text="Additional Info: Ensure you have access to the email account associated with your user profile to receive the reset link.",
            doc_id="KB001.md", metadata={"title": "Password Reset Guide", "category": "Accounts"}, distance=0.85
        )
    ]
    return QueryResponse(
        generated_answer=sample_answer, source_chunks=sample_chunks,
        session_id="demo_conv_67890", message="This is a demo response."
    )

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    llm_status = "Not initialized"
    if LLMGenerator._model and LLMGenerator._tokenizer:
        llm_status = f"Initialized: {LLMGenerator._model_name_loaded} on {LLMGenerator._device} (model device: {LLMGenerator._model.device})"
    elif LLMGenerator._model_name_loaded :
        llm_status = f"Initialization failed for {LLMGenerator._model_name_loaded}"

    return {
        "message": "Welcome to the AI Tech Support Agent API.",
        "docs_url": "/docs", "redoc_url": "/redoc",
        "vector_store_status": {
            "initialized": vector_store_manager.is_initialized() if vector_store_manager else False,
            "item_count": vector_store_manager.get_ntotal_faiss() if vector_store_manager else 0,
            "document_count": vector_store_manager.get_document_count() if vector_store_manager else 0,
            "dimension": vector_store_manager.get_dimension() if vector_store_manager else None,
        },
        "embedding_model_status": {
            "model_name": EmbeddingModel._model_name, "device": EmbeddingModel._device,
            "initialized": EmbeddingModel._model is not None
        },
        "llm_status": llm_status,
        "configured_embedding_model": config.MODEL_NAME,
        "configured_embedding_dimension": config.EMBEDDING_DIMENSION,
        "configured_llm_model": config.LLM_MODEL_NAME,
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server for local development on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        app, host=config.API_HOST, port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
```

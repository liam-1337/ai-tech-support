# AI Tech Support Agent - User Guide

## 1. Introduction
The AI Tech Support Agent is a sophisticated application designed to provide intelligent assistance for technical support queries. It leverages modern AI techniques, including Natural Language Processing (NLP) and Large Language Models (LLMs), to understand user questions and provide relevant answers based on a knowledge base of documents and past support tickets. Its primary purpose is to offer quick, accurate, and context-aware support, reducing response times and improving user satisfaction. This system is intended for IT support teams, managed service providers, and end-users who need access to a well-curated technical knowledge base.

## 2. System Architecture
- High-level overview: Backend (Python/FastAPI, RAG pipeline) and Frontend (Next.js).
- Diagram (placeholder for now, e.g., "[High-level Architecture Diagram]")
The system follows a Retrieval Augmented Generation (RAG) architecture.
-   **Backend (Python/FastAPI):**
    -   Manages data ingestion pipelines to process documents and tickets.
    -   Uses sentence transformers to create vector embeddings of text chunks.
    -   Stores embeddings in a FAISS vector store for efficient similarity searching.
    -   When a query is received, it retrieves relevant chunks from the vector store.
    -   Uses a Large Language Model (LLM) to generate a coherent answer based on the retrieved chunks and the user's query.
    -   Exposes RESTful APIs for the frontend.
-   **Frontend (Next.js/TypeScript):**
    -   Provides a user-friendly chat interface.
    -   Sends user queries to the backend API.
    -   Displays the AI-generated answers and the source text chunks that informed the answer.

## 3. Features
-   **Querying knowledge base via chat:** Users can ask questions in natural language through an intuitive web interface.
-   **Retrieval of source documents/chunks:** Alongside answers, the system shows the specific text excerpts from the knowledge base used by the AI, ensuring transparency and allowing users to verify information.
-   **Data ingestion from general documents:** Easily build the knowledge base by ingesting various document formats (PDF, DOCX, PPTX, TXT) from a specified directory.
-   **Data ingestion and processing from ticket data (JSON):** Augment the knowledge base with structured data from past support tickets, including problem descriptions and resolution steps, enabling the AI to learn from historical solutions.

## 4. Setup and Installation
### 4.1. Prerequisites
- Python (3.9+ recommended)
- Node.js (18.x or 20.x recommended)
- Access to a terminal/shell.
- `pip` for Python packages.
- `npm` or `yarn` for Node.js packages.

### 4.2. Backend Setup
- Clone repository: `git clone <repository_url>`
- Navigate to project root.
- Python virtual environment:
  - `python -m venv venv`
  - `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r ai_tech_support_agent/requirements.txt`
- Configuration (`.env` file for backend, located at the project root):
  - Copy `.env.example` to `.env` in the project root.
  - Explain key variables:
    - `MODEL_NAME`: Sentence Transformer model for embeddings.
    - `EMBEDDING_DIMENSION`: Dimension of embeddings (must match `MODEL_NAME`).
    - `LLM_MODEL_NAME`: Hugging Face model for answer generation.
    - `LLM_PROMPT_TEMPLATE`: Template for the prompt sent to the LLM.
    - `VECTOR_STORE_INDEX_DIR`: Path to store FAISS index.
    - `API_HOST`, `API_PORT`: Backend server address.
- Running the backend (development): `uvicorn ai_tech_support_agent.app.main:app --host 0.0.0.0 --port 8000 --reload` (from the project root directory).

### 4.3. Frontend Setup
- Navigate to `frontend` directory: `cd frontend`
- Install dependencies: `npm install` (or `yarn install`)
- Configuration:
  - Copy `frontend/.env.example` to `frontend/.env.local` (standard Next.js practice for local environment variables).
  - Explain key variables:
    - `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` (adjust if your backend runs on a different URL).
- Running the frontend: `npm run dev` (or `yarn dev`).

## 5. Data Ingestion
### 5.1. General Documents
- Script: `python scripts/ingest_documents.py`
- Arguments:
  - `--input-dir`: Directory with documents.
  - `--chunk-strategy`: `token_count`, `paragraph`, `sentence`.
  - `--max-tokens`, `--overlap-tokens`, `--sentences-per-chunk`: Strategy-specific params.
  - `--recreate-index`: Boolean flag.
- Supported types: .pdf, .docx, .pptx, .txt.
- Note: The script recursively scans the input directory for files with these extensions.
- Example: `python scripts/ingest_documents.py --input-dir ./path/to/your/docs --chunk-strategy token_count`

### 5.2. Ticket Data
- Script: `python ai_tech_support_agent/scripts/ingest_tickets.py`
- Input format: JSON file. Describe expected fields:
  - `ticket_id`: (String) Unique identifier for the ticket. *Required, cannot be empty.*
  - `creation_date`: (String) ISO format date/time of ticket creation. *Optional.*
  - `status`: (String) Current status of the ticket (e.g., "Resolved", "Closed"). *Required, cannot be empty.*
  - `problem_description`: (String) Detailed description of the issue. *Required, cannot be empty.*
  - `resolution_steps`: (String) Steps taken to resolve the issue. *Required, cannot be empty.*
  - `category`: (String) Category of the ticket (e.g., "Network", "Software"). *Optional.*
  - `tags`: (List of Strings) Relevant tags. *Optional.*
  - `affected_software`: (List of Strings) Software involved. *Optional.*
  - `affected_hardware`: (List of Strings) Hardware involved. *Optional.*
- Note: The input must be a JSON file containing a list of ticket objects matching this structure.
- Arguments:
  - `--input-file`: Path to JSON ticket data.
  - `--chunk-strategy`, etc. (similar to document ingestion).
  - `--recreate-index`.
- Example: `python ai_tech_support_agent/scripts/ingest_tickets.py --input-file ./path/to/tickets.json`

## 6. Using the Application
- Accessing the chat: Open browser to frontend URL (e.g., `http://localhost:3000`).
- Asking questions.
- Understanding responses:
  When you receive a response from the AI, it will typically include:
  -   **Generated Answer:** The AI's direct answer to your question.
  -   **Source Chunks:** These are snippets of text extracted from the ingested documents or tickets that the AI considered most relevant to your query and used to formulate its answer. Reviewing these can provide additional context or verify the information.

## 7. Configuration Details
- `.env` file at project root for backend.
- `ai_tech_support_agent/app/config.py` (how it loads `.env` and defaults).
- Key backend config variables (reiterate with more detail if needed):
  - `MODEL_NAME`, `EMBEDDING_DIMENSION`
  - `LLM_MODEL_NAME`, `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`, `LLM_PROMPT_TEMPLATE`
  - `VECTOR_STORE_INDEX_DIR`, `TOP_K_RESULTS`
  - `LOG_LEVEL`
- Frontend configuration (`frontend/.env.local`):
  - `NEXT_PUBLIC_API_BASE_URL`

## 8. Troubleshooting
- **Models not downloading:** Ensure you have a stable internet connection. Some models are large and may take time. Check Hugging Face Hub status if issues persist. Ensure sufficient disk space in the cache directory (usually `~/.cache/huggingface/hub`).
- **FAISS/vector store errors:** (dimension mismatches, file permissions for `VECTOR_STORE_INDEX_DIR`).
- **Backend/Frontend connection issues:** (API URL, CORS if applicable).
- **Python/Node.js dependency conflicts.**
- **`ModuleNotFoundError` for project modules:** Ensure you are running commands from the project root directory and the Python virtual environment is activated.
- **Frontend shows 'Error connecting to backend' or similar:** Verify the backend server is running and accessible at the URL specified in `frontend/.env.local` (as `NEXT_PUBLIC_API_BASE_URL`). Check for CORS errors in the browser console if the backend is running but still inaccessible.
- **Incorrect embedding dimensions:** If you change `MODEL_NAME`, ensure `EMBEDDING_DIMENSION` in your `.env` file matches the new model's output dimension. You might need to delete and recreate the vector store index if dimensions are incompatible.

## 9. Development (Optional)
-   **Backend `ai_tech_support_agent/app/logic/`**:
    *   `document_loader.py`: Handles loading and extracting text from various file formats.
    *   `ticket_loader.py`: Parses and validates JSON ticket data.
    *   `text_processor.py`: Responsible for text cleaning and chunking strategies.
    *   `embedding_generator.py`: Manages sentence transformer models and generates embeddings.
    *   `vector_store_manager.py`: Handles FAISS index creation, saving, loading, and searching.
    *   `llm_handler.py`: Manages the LLM for generating answers based on context.
    *   `draft_generator.py`: Generates initial knowledge base articles from ticket data.
    *   `ticket_analyzer.py`: Uses an LLM to summarize or extract key info from tickets.
-   **Frontend `frontend/src/`**:
    *   `app/`: Main Next.js application pages and layout.
    *   `components/`: Reusable UI components (e.g., ChatWindow, MessageBubble).
    *   `context/`: React Context for global state management (e.g., ChatContext).
    *   `utils/`: Utility functions, including `apiClient.ts` for backend communication.
    *   `types/`: TypeScript type definitions.

## 10. Future Enhancements
-   Support for a wider range of document formats (e.g., HTML, CSV).
-   A web-based UI for managing data ingestion processes.
-   User authentication and role-based access control.
-   Advanced analytics on queries and AI performance.
-   Integration with real-time ticketing systems.

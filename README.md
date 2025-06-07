# AI Tech Support Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The AI Tech Support Agent is an intelligent application designed to provide assistance for technical support queries using AI, including Natural Language Processing and Large Language Models. It features a RAG (Retrieval Augmented Generation) pipeline to deliver context-aware answers from a knowledge base built from documents and support tickets.

## Key Features

*   **Conversational AI Support:** Ask questions in natural language via a web-based chat interface.
*   **Contextual Answers with Sources:** Receive answers generated from your knowledge base, complete with references to the source documents.
*   **Flexible Data Ingestion:**
    *   Load general documents (`.pdf`, `.docx`, `.pptx`, `.txt`).
    *   Process and learn from historical support ticket data (JSON format).
*   **Configurable AI Models:** Customize embedding and language models to suit your needs.

## Quick Start

This guide provides a basic setup. For detailed instructions, please see the **[Full User Guide](USER_GUIDE.md)**.

### Prerequisites

*   Python 3.9+
*   Node.js 18.x+
*   Git

### 1. Clone the Repository

```bash
git clone <your_repository_url>
cd ai-tech-support # Or your repository's root folder name
```

### 2. Backend Setup (AI & API)

```bash
# Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r ai_tech_support_agent/requirements.txt

# Configure backend (copy .env.example to .env and edit)
cp .env.example .env
# nano .env  # Or use your preferred editor to set model names, API keys, etc.

# (Optional) Ingest your data - see USER_GUIDE.md for details
# python scripts/ingest_documents.py --input-dir path/to/your/docs
# python ai_tech_support_agent/scripts/ingest_tickets.py --input-file path/to/your/tickets.json

# Run the backend server
uvicorn ai_tech_support_agent.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API will be available at `http://localhost:8000`.

### 3. Frontend Setup (Chat Interface)

```bash
# Navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install

# Configure frontend (copy .env.example to .env.local and edit if needed)
cp .env.example .env.local
# nano .env.local # Set NEXT_PUBLIC_API_BASE_URL if different from http://localhost:8000

# Run the frontend development server
npm run dev
```
The chat interface will be available at `http://localhost:3000`.

## Configuration

*   **Backend:** Uses a `.env` file in the project root (see `.env.example`). Key settings include Hugging Face model names for embeddings and LLMs, vector store paths, and API configurations.
*   **Frontend:** Uses an `.env.local` file in the `frontend` directory (see `frontend/.env.example`). Key settings include the backend API URL.

Refer to the [User Guide section on Configuration](USER_GUIDE.md#7-configuration-details) for more information.

## Technologies Used

*   **Backend:** Python, FastAPI, Hugging Face Transformers, Sentence Transformers, FAISS, Pydantic
*   **Frontend:** Next.js, React, TypeScript, Tailwind CSS
*   **Data Ingestion:** Support for PDF, DOCX, PPTX, TXT, and structured JSON tickets.

## Full Documentation

For detailed information on architecture, setup, data ingestion, advanced configuration, troubleshooting, and development, please refer to the **[USER_GUIDE.md](USER_GUIDE.md)**.

## Contributing

Contributions are welcome! Please refer to the development section in the User Guide and consider opening an issue or pull request.

## License



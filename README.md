# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`



## How Code Works

This section is written in ASD-STE100 (Simplified Technical English).

### What the system does

This system is a chatbot. It answers questions about course materials. It has a Python backend (FastAPI) and a web page (plain JavaScript).

### Main parts

| File | Function |
|---|---|
| `app.py` | Receives HTTP requests. Serves the web page. |
| `rag_system.py` | Controls the other parts. |
| `ai_generator.py` | Sends requests to Claude. |
| `search_tools.py` | Gives Claude a tool to search the courses. |
| `vector_store.py` | Stores and searches the course text (ChromaDB). |
| `document_processor.py` | Reads course files and splits them into chunks. |
| `session_manager.py` | Keeps the chat history in memory. |
| `config.py` | Holds all settings. |

You must start the server from the `backend/` folder. The code uses relative paths.

### What happens when a user asks a question

1. The browser sends the question to `POST /api/query`.
2. `RAGSystem` creates a session if the request has none.
3. `AIGenerator` sends the question to Claude. The request includes the system prompt, the chat history, and the tool definitions.
4. Claude decides if it needs to search.
   - If the question is general, Claude answers directly.
   - If the question is about a course, Claude calls `search_course_content`.
5. If Claude calls the tool, the system runs the search in ChromaDB.
6. The system sends the search results to Claude in a second request. This second request has no tools. Therefore, the system allows only one search for each question.
7. Claude writes the final answer.
8. `RAGSystem` collects the sources from the tool and sends the answer and sources to the browser.

### How the data is stored

ChromaDB has two collections. Both use the `all-MiniLM-L6-v2` model to make embeddings.

- **`course_catalog`**: One entry for each course. The ID is the course title. The system uses this collection to find the exact course name when a user types a partial name.
- **`course_content`**: Text chunks. Each chunk has the course title, the lesson number, and the chunk index.

### How the system loads course files

When the server starts, it reads each file in the `docs/` folder. The file format is:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson text>
```

The system splits the text into chunks of 800 characters. Each chunk overlaps the previous chunk by 100 characters.

### Important limits

- The system skips a course if its title is already in the database. If you change a course file, delete `backend/chroma_db` to load it again.
- Chat history is in memory only. The history is lost when the server stops.
- Claude can write at most 800 tokens in one answer.
- The system treats PDF and DOCX files as plain text. It does not parse them.

### Tests

- Run `cd backend && uv run pytest tests` for the automated tests. They do not use the network.
- Run `cd backend && uv run python live_checks.py` to check the real setup. Use this when you see "Query failed".

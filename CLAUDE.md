# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install deps: `uv sync` (Python >= 3.13, managed with `uv`; always use `uv run` / `uv add`, not pip)
- Run the app: `./run.sh`, or `cd backend && uv run uvicorn app:app --reload --port 8000`
  - Web UI: http://localhost:8000 — API docs: http://localhost:8000/docs
- Requires `ANTHROPIC_API_KEY` in a root `.env` file (see `.env.example`).

There is no test suite, linter, or formatter configured.

The server **must be started from `backend/`**: `app.py` uses relative paths (`../docs`, `../frontend`) and ChromaDB persists to `./chroma_db` (i.e. `backend/chroma_db`).

## Architecture

Course-materials RAG chatbot: FastAPI backend + vanilla JS frontend (`frontend/`, served as static files by the backend at `/`). Backend modules in `backend/` import each other as top-level modules (no package), so run from that directory.

### Query flow (tool-based RAG, not retrieve-then-generate)

1. `POST /api/query` (`app.py`) → `RAGSystem.query()` (`rag_system.py`), creating a session if none was sent.
2. `AIGenerator.generate_response()` (`ai_generator.py`) calls Claude with the system prompt, conversation history (embedded in the system prompt as text), and tool definitions from `ToolManager`.
3. Claude decides whether to call the `search_course_content` tool (`CourseSearchTool` in `search_tools.py`). General-knowledge questions are answered without searching.
4. If `stop_reason == "tool_use"`, `_handle_tool_execution` runs the tools and makes **one** follow-up API call *without* tools — so only a single round of tool use is supported per query.
5. Sources are passed back out-of-band: `CourseSearchTool` stores `last_sources` on itself during `execute`; `RAGSystem` reads them via `ToolManager.get_last_sources()` and then calls `reset_sources()`.
6. `SessionManager` (in-memory, lost on restart) keeps the last `MAX_HISTORY` exchanges per session.

`GET /api/courses` returns course count/titles from the catalog.

### Vector store (`vector_store.py`)

Two ChromaDB collections, both embedded with `all-MiniLM-L6-v2` via sentence-transformers:
- `course_catalog` — one doc per course, **ID = course title**; lesson list stored as a JSON string in metadata. Used to fuzzy-resolve a user-supplied course name to an exact title (`_resolve_course_name` does a 1-result semantic search).
- `course_content` — text chunks with `course_title`, `lesson_number`, `chunk_index` metadata; IDs are `{title_with_underscores}_{chunk_index}`.

`VectorStore.search()` resolves the course name, builds a `where` filter (course and/or lesson), then queries `course_content`.

### Ingestion (`document_processor.py`)

On startup, `app.py` loads every `.txt/.pdf/.docx` in `docs/` via `RAGSystem.add_course_folder(clear_existing=False)`. Courses whose title already exists in the catalog are skipped, so **editing a doc under an existing title won't re-ingest it** — delete `backend/chroma_db` (or pass `clear_existing=True`) to rebuild. (Note: files are read as plain text; PDF/DOCX aren't actually parsed.)

Expected document format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>
Lesson 1: ...
```
Text is split into sentence-based chunks (`CHUNK_SIZE` 800 chars, `CHUNK_OVERLAP` 100). Quirk: chunk context prefixes are inconsistent — for most lessons only the first chunk gets `"Lesson N content: "`, while every chunk of the final lesson gets `"Course <title> Lesson N content: "`.

### Configuration

All tunables (model, embedding model, chunk sizes, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`) live in the `Config` dataclass in `backend/config.py`. Claude response length is capped by `max_tokens: 800` in `AIGenerator`, and the assistant's behavior (one search per query, no meta-commentary) is governed by `AIGenerator.SYSTEM_PROMPT`.

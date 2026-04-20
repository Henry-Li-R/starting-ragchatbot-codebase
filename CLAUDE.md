# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

UI: `http://localhost:8000` — API docs: `http://localhost:8000/docs`

Install dependencies: `uv sync`

Always use `uv run` / `uv sync` — never pip or bare python/uvicorn invocations.

## Architecture

Full-stack RAG chatbot. FastAPI backend (`backend/`) serves both the API and the static frontend (`frontend/`). ChromaDB persists to `./chroma_db` (relative to `backend/`). Course `.txt` files live in `docs/` and are loaded into ChromaDB on startup (skipped if already present).

**Query flow:**
1. `POST /api/query` → `RAGSystem.query()` → first Claude API call with `search_course_content` tool available
2. If Claude invokes the tool: `VectorStore.search()` embeds the query via `all-MiniLM-L6-v2` and retrieves top-5 chunks from ChromaDB
3. Tool results injected → second Claude API call (no tools) produces the final answer
4. Conversation history is kept in-memory per session (capped at `MAX_HISTORY * 2 = 4` messages) and injected into the system prompt as plain text

**Document ingestion flow:**
Course `.txt` files must follow this format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content...>
```
`DocumentProcessor` splits lesson content into sentence-boundary-respecting chunks (`CHUNK_SIZE=800` chars, `CHUNK_OVERLAP=100`). Each chunk is stored in the `course_content` ChromaDB collection; course metadata goes into `course_catalog`.

**Key config** (`backend/config.py`):
- `ANTHROPIC_MODEL`: `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2`
- `CHUNK_SIZE`: 800, `CHUNK_OVERLAP`: 100, `MAX_RESULTS`: 5, `MAX_HISTORY`: 2

## Two ChromaDB Collections

- `course_catalog` — one document per course (title text + metadata); used for fuzzy course-name resolution when the tool is called with a `course_name`
- `course_content` — one document per chunk; filtered by `course_title` and/or `lesson_number` metadata fields

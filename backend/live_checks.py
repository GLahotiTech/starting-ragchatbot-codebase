"""Manual smoke checks for the LIVE setup (real DB, real config, real LLM endpoint).

Not a pytest file (the name doesn't match test_*.py, so `pytest` never collects it).
Run from backend/:

    uv run python live_checks.py              # run every check
    uv run python live_checks.py db endpoint  # run only the named checks

Checks never modify your data: the DB checks operate on a temporary COPY of
backend/chroma_db. The `endpoint` check makes one real 10-token API request.
Exit code is 0 if all selected checks pass, 1 otherwise.
"""

import json
import os
import shutil
import sys
import tempfile

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from config import config  # noqa: E402
from search_tools import CourseSearchTool  # noqa: E402
from vector_store import VectorStore  # noqa: E402


def _open_db_copy(tmp_dir):
    """Open a VectorStore over a copy of backend/chroma_db (never touches the original)."""
    src = os.path.join(BACKEND_DIR, "chroma_db")
    if not os.path.isdir(src):
        raise AssertionError(
            "backend/chroma_db does not exist (start the server once to ingest docs)"
        )
    dst = os.path.join(tmp_dir, "chroma_copy")
    shutil.copytree(src, dst)
    return VectorStore(dst, config.EMBEDDING_MODEL, config.MAX_RESULTS)


def check_db_has_content(store):
    """Both collections are populated; an empty DB would make every search fail."""
    courses = store.get_course_count()
    chunks = store.course_content.count()
    assert courses > 0, "course_catalog is empty"
    assert chunks > 0, "course_content is empty"
    return f"{courses} courses, {chunks} chunks"


def check_search_returns_content(store):
    """An unfiltered search returns content, not an error or 'no content'."""
    out = CourseSearchTool(store).execute("what is covered in the lessons")
    assert not out.startswith("Search error"), out
    assert not out.startswith("No relevant content"), out
    return f"{len(out)} chars returned"


def check_course_filtered_search(store):
    """Filtering by a real course title works."""
    title = store.get_existing_course_titles()[0]
    out = CourseSearchTool(store).execute("introduction", course_name=title)
    assert not out.startswith(
        ("Search error", "No course found", "No relevant content")
    ), out
    return title


def check_course_and_lesson_filtered_search(store):
    """Filtering by a real course title AND a lesson that exists in it works."""
    meta = store.get_all_courses_metadata()[0]
    title, lesson = meta["title"], meta["lessons"][0]["lesson_number"]
    out = CourseSearchTool(store).execute(
        "introduction", course_name=title, lesson_number=lesson
    )
    assert not out.startswith(
        ("Search error", "No course found", "No relevant content")
    ), out
    return f"{title} / lesson {lesson}"


def check_config():
    """ANTHROPIC_API_KEY is set (environment or .env)."""
    assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is empty"
    return f"model={config.ANTHROPIC_MODEL} base_url={config.ANTHROPIC_BASE_URL}"


def check_endpoint():
    """Send one real 10-token request to the configured endpoint and model (costs a few tokens)."""
    import anthropic

    assert config.ANTHROPIC_API_KEY, "no API key, cannot call the endpoint"
    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY,
        base_url=config.ANTHROPIC_BASE_URL,
        timeout=20,
        max_retries=0,
    )
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.content, "endpoint answered but returned empty content"
    return "reachable, model accepted"


# name -> (function, needs the DB copy)
CHECKS = {
    "db": (check_db_has_content, True),
    "search": (check_search_returns_content, True),
    "course": (check_course_filtered_search, True),
    "lesson": (check_course_and_lesson_filtered_search, True),
    "config": (check_config, False),
    "endpoint": (check_endpoint, False),
}


def main(names):
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"Unknown check(s): {unknown}. Available: {list(CHECKS)}")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = None
        for name in names or CHECKS:
            fn, needs_db = CHECKS[name]
            try:
                if needs_db:
                    if store is None:
                        store = _open_db_copy(tmp_dir)
                    detail = fn(store)
                else:
                    detail = fn()
                print(f"PASS  {name:9} {detail}")
            except Exception as e:  # noqa: BLE001 - report any failure, keep going
                failures += 1
                print(f"FAIL  {name:9} {type(e).__name__}: {e}")
    print(f"\n{len(names or CHECKS) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

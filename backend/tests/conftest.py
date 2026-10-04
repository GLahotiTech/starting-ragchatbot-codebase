"""Shared pytest setup.

pytest loads this file automatically. Any function decorated with @pytest.fixture
here can be used in ANY test file in this folder: just add a parameter with the
same name as the fixture to the test function and pytest passes in its return value.
No import is needed (plain helper functions, like text_response, do need importing).
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Backend modules are top-level (no package), so put backend/ on the import path.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import Config  # noqa: E402
from models import Course, CourseChunk, Lesson  # noqa: E402
from vector_store import VectorStore  # noqa: E402


@pytest.fixture
def sample_course():
    """A tiny made-up course ("Intro to Widgets", 2 lessons) plus one chunk per lesson.

    Returns a (Course, [CourseChunk, ...]) tuple. Tests use it as known data to search.
    """
    course = Course(
        title="Intro to Widgets",
        course_link="https://example.com/widgets",
        instructor="Ada",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Overview",
                lesson_link="https://example.com/widgets/0",
            ),
            Lesson(
                lesson_number=1,
                title="Building",
                lesson_link="https://example.com/widgets/1",
            ),
        ],
    )
    chunks = [
        CourseChunk(
            content="Lesson 0 content: Widgets are small reusable components.",
            course_title=course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Lesson 1 content: To build a widget, attach the gizmo to the sprocket.",
            course_title=course.title,
            lesson_number=1,
            chunk_index=1,
        ),
    ]
    return course, chunks


@pytest.fixture
def second_course():
    """A second made-up course ("Gadget Basics"), so tests can tell courses apart.

    With a single course in the DB, course-name resolution and filtering succeed
    trivially; a second course makes those tests meaningful.
    """
    course = Course(
        title="Gadget Basics",
        course_link="https://example.com/gadgets",
        instructor="Bo",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Power",
                lesson_link="https://example.com/gadgets/0",
            ),
            Lesson(
                lesson_number=1,
                title="Repair",
                lesson_link="https://example.com/gadgets/1",
            ),
        ],
    )
    chunks = [
        CourseChunk(
            content="Lesson 0 content: Gadgets whir and click when powered by a battery.",
            course_title=course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Lesson 1 content: To repair a gadget, replace the flux capacitor.",
            course_title=course.title,
            lesson_number=1,
            chunk_index=1,
        ),
    ]
    return course, chunks


@pytest.fixture
def real_store(tmp_path, sample_course, second_course):
    """Real ChromaDB + real embedding model in a temp dir, seeded with BOTH sample courses.

    `tmp_path` is a built-in pytest fixture: a fresh empty directory per test, so
    tests never share or pollute data.
    """
    store = VectorStore(str(tmp_path / "chroma"), Config.EMBEDDING_MODEL, max_results=5)
    for course, chunks in (sample_course, second_course):
        store.add_course_metadata(course)
        store.add_course_content(chunks)
    return store


# ---- Anthropic response fakes -------------------------------------------------
# These mimic the shape of anthropic SDK responses (only the attributes the code reads).
def text_response(text="final answer"):
    """Fake Claude reply that is finished and contains one text block."""
    return SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)]
    )


def tool_use_response(name="search_course_content", tool_input=None, tool_id="toolu_1"):
    """Fake Claude reply that asks us to run a tool (stop_reason == "tool_use")."""
    block = SimpleNamespace(
        type="tool_use", name=name, id=tool_id, input=tool_input or {"query": "widgets"}
    )
    return SimpleNamespace(stop_reason="tool_use", content=[block])


@pytest.fixture
def api(monkeypatch):
    """Yields (TestClient, fake_rag): the real app.py wired to a fake RAGSystem.

    app.py builds a RAGSystem at import time and mounts ../frontend, so RAGSystem is
    patched before a fresh import, and cwd is backend/. TestClient is used WITHOUT a
    `with` block so the startup (doc ingestion) hook never runs.
    """
    from fastapi.testclient import TestClient

    fake_rag = MagicMock()
    monkeypatch.setattr("rag_system.RAGSystem", lambda cfg: fake_rag)
    monkeypatch.chdir(BACKEND_DIR)
    sys.modules.pop("app", None)
    import app
    yield TestClient(app.app), fake_rag
    sys.modules.pop("app", None)


@pytest.fixture
def mock_anthropic(monkeypatch):
    """Replace anthropic.Anthropic with a fake so AIGenerator never touches the network.

    Returns the fake client (a MagicMock). Tests configure what it replies with, e.g.
        mock_anthropic.messages.create.return_value = text_response("hi")
        mock_anthropic.messages.create.side_effect = [reply1, reply2]  # one per call
    and inspect what was sent via .call_args / .call_args_list.
    `monkeypatch` (built into pytest) undoes the patch automatically after each test.
    """
    client = MagicMock()
    monkeypatch.setattr("ai_generator.anthropic.Anthropic", lambda **kw: client)
    return client

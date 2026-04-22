"""
test_course_search_tool.py
==========================
Unit tests for CourseSearchTool.execute() in backend/search_tools.py.

WHAT IS BEING TESTED
--------------------
CourseSearchTool is the bridge between Claude and ChromaDB. When Claude decides
it needs course content it calls this tool, which:
  1. Asks the VectorStore to search ChromaDB for relevant text chunks.
  2. Formats those chunks into a labelled string Claude can read.
  3. Records "sources" (course + lesson metadata) so the UI can show citations.

HOW THESE TESTS WORK (mocking explained)
-----------------------------------------
We never talk to a real database here. Instead every test creates a
MagicMock — a fake VectorStore we fully control. We tell the fake store
exactly what to return, then check that CourseSearchTool behaves correctly.

This is called "unit testing with mocks": isolate one component, fake its
dependencies, and verify only its own logic.

PYTEST QUICK REFERENCE
-----------------------
- Any function whose name starts with test_ is auto-discovered and run.
- `assert <expr>` fails the test when the expression is False.
- @pytest.fixture defines reusable setup (runs fresh for each test).
- Class names starting with Test group related tests (cosmetic only).
- Run: uv run pytest backend/tests/test_course_search_tool.py -v

THREE BUGS THIS SUITE EXPOSES
-------------------------------
Bug 1a – last_sources not cleared after empty results (search_tools.py:77-83)
Bug 1b – last_sources not cleared after a search error  (search_tools.py:73-74)
Bug 2  – `if lesson_number:` is falsy for 0, dropping lesson 0 from the
          no-results message                             (search_tools.py:81)
"""

import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

def make_results(docs, metas, error=None):
    """
    Build a SearchResults without touching ChromaDB.

    Args:
        docs  – list of document strings (chunk text the AI will read)
        metas – list of metadata dicts, one per document
        error – optional error string (simulates a DB failure)
    """
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
        error=error,
    )


@pytest.fixture
def mock_store():
    """
    A fake VectorStore. MagicMock auto-creates any method you call on it.
    We pre-set get_lesson_link to return None so tests that don't care about
    URLs don't have to configure it themselves.
    """
    store = MagicMock()
    store.get_lesson_link.return_value = None
    return store


@pytest.fixture
def tool(mock_store):
    """CourseSearchTool wired to the fake store."""
    return CourseSearchTool(mock_store)


# ---------------------------------------------------------------------------
# Successful searches — output formatting
# ---------------------------------------------------------------------------

class TestSuccessfulSearch:

    def test_output_contains_course_lesson_header(self, tool, mock_store):
        """
        Each result block must start with [CourseName - Lesson N] so the AI
        knows where each piece of text came from.
        """
        mock_store.search.return_value = make_results(
            docs=["Chroma is a vector database."],
            metas=[{"course_title": "RAG Course", "lesson_number": 2}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson2"

        result = tool.execute(query="what is chroma")

        assert "[RAG Course - Lesson 2]" in result
        assert "Chroma is a vector database." in result

    def test_header_has_no_lesson_when_metadata_lacks_lesson_number(self, tool, mock_store):
        """
        When a chunk has no lesson_number in its metadata the header should be
        just [CourseName], NOT [CourseName - Lesson None].
        """
        mock_store.search.return_value = make_results(
            docs=["Some content."],
            metas=[{"course_title": "RAG Course"}],
        )

        result = tool.execute(query="test")

        assert "[RAG Course]" in result
        assert "Lesson" not in result

    def test_all_result_blocks_appear_in_output(self, tool, mock_store):
        """Multiple results are all present in the output string."""
        mock_store.search.return_value = make_results(
            docs=["doc A", "doc B"],
            metas=[
                {"course_title": "Course X", "lesson_number": 1},
                {"course_title": "Course X", "lesson_number": 2},
            ],
        )

        result = tool.execute(query="test")

        assert "doc A" in result and "doc B" in result
        assert "[Course X - Lesson 1]" in result
        assert "[Course X - Lesson 2]" in result

    def test_search_called_with_all_provided_arguments(self, tool, mock_store):
        """
        execute() must forward query, course_name, and lesson_number to
        store.search() unchanged — it must not silently drop filters.
        """
        mock_store.search.return_value = make_results([], [])

        tool.execute(query="embeddings", course_name="MCP", lesson_number=3)

        mock_store.search.assert_called_once_with(
            query="embeddings", course_name="MCP", lesson_number=3
        )


# ---------------------------------------------------------------------------
# Sources tracking
# ---------------------------------------------------------------------------

class TestSourcesTracking:
    """
    After execute() the tool.last_sources list must hold one dict per result,
    with keys 'label' (human-readable string) and 'url' (lesson link or None).
    The RAGSystem reads this list to populate the UI citation panel.
    """

    def test_last_sources_contains_label_and_url(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://mcp.example.com/1"

        tool.execute(query="test")

        assert tool.last_sources == [
            {"label": "MCP Course - Lesson 1", "url": "https://mcp.example.com/1"}
        ]

    def test_last_sources_url_is_none_when_no_lesson_number(self, tool, mock_store):
        """
        No lesson_number → can't look up a link → url must be None and
        get_lesson_link must NOT be called at all.
        """
        mock_store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course"}],
        )

        tool.execute(query="test")

        assert tool.last_sources[0]["url"] is None
        mock_store.get_lesson_link.assert_not_called()

    def test_last_sources_count_matches_result_count(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            docs=["a", "b", "c"],
            metas=[
                {"course_title": "C", "lesson_number": 1},
                {"course_title": "C", "lesson_number": 2},
                {"course_title": "C", "lesson_number": 3},
            ],
        )

        tool.execute(query="test")

        assert len(tool.last_sources) == 3

    def test_get_lesson_link_called_once_per_result_with_lesson(self, tool, mock_store):
        mock_store.search.return_value = make_results(
            docs=["a", "b"],
            metas=[
                {"course_title": "C", "lesson_number": 1},
                {"course_title": "C", "lesson_number": 3},
            ],
        )

        tool.execute(query="test")

        assert mock_store.get_lesson_link.call_count == 2


# ---------------------------------------------------------------------------
# Empty and error result handling
# ---------------------------------------------------------------------------

class TestEmptyAndErrorResults:

    def test_empty_results_returns_no_content_message(self, tool, mock_store):
        mock_store.search.return_value = make_results(docs=[], metas=[])

        result = tool.execute(query="xyz")

        assert "No relevant content found" in result

    def test_empty_with_course_filter_names_the_course(self, tool, mock_store):
        """When filtering to a course, the course name must appear in the message."""
        mock_store.search.return_value = make_results(docs=[], metas=[])

        result = tool.execute(query="xyz", course_name="MCP Course")

        assert "in course 'MCP Course'" in result

    def test_empty_with_lesson_filter_names_the_lesson(self, tool, mock_store):
        mock_store.search.return_value = make_results(docs=[], metas=[])

        result = tool.execute(query="xyz", lesson_number=5)

        assert "in lesson 5" in result

    def test_search_error_returns_the_error_string(self, tool, mock_store):
        """DB errors must bubble up so Claude can tell the user what went wrong."""
        mock_store.search.return_value = make_results(
            docs=[], metas=[], error="Connection timeout"
        )

        result = tool.execute(query="test")

        assert "Connection timeout" in result


# ---------------------------------------------------------------------------
# Bug-exposure tests  ← THESE WILL FAIL before fixes are applied
# ---------------------------------------------------------------------------

class TestBugsExposed:
    """
    Each test below is written to FAIL against the current code, then PASS
    after the corresponding fix is applied.

    How to read a failure:
      pytest prints "AssertionError" followed by the message string we provide.
      That message explains the bug and the fix.
    """

    def test_bug1a_last_sources_cleared_after_empty_results(self, tool, mock_store):
        """
        BUG 1a — last_sources not reset when results are empty.

        Root cause (search_tools.py:77-83):
            execute() returns early when is_empty() is True, before
            _format_results() is called. _format_results() is the ONLY place
            that writes self.last_sources, so the old value persists.

        Impact: after a successful search followed by an empty one, the UI
        will still show the first search's citations — wrong and misleading.

        Fix: add `self.last_sources = []` before the empty-results early return.
        """
        # First call — populates last_sources
        mock_store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com"
        tool.execute(query="first")
        assert len(tool.last_sources) == 1  # sanity check

        # Second call returns nothing — last_sources must be cleared
        mock_store.search.return_value = make_results(docs=[], metas=[])
        tool.execute(query="second")

        assert tool.last_sources == [], (
            "last_sources should be [] after empty results but stale sources "
            "from the previous call persist. "
            "Fix: add `self.last_sources = []` before the empty-results return."
        )

    def test_bug1b_last_sources_cleared_after_error_results(self, tool, mock_store):
        """
        BUG 1b — same stale-sources problem on the error path.

        Fix: add `self.last_sources = []` before the error early return.
        """
        mock_store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP", "lesson_number": 1}],
        )
        mock_store.get_lesson_link.return_value = "https://example.com"
        tool.execute(query="first")

        mock_store.search.return_value = make_results(docs=[], metas=[], error="DB error")
        tool.execute(query="second")

        assert tool.last_sources == [], (
            "last_sources should be [] after a search error but stale sources persist. "
            "Fix: add `self.last_sources = []` before the error early return."
        )

    def test_bug2_lesson_zero_appears_in_no_results_message(self, tool, mock_store):
        """
        BUG 2 — `if lesson_number:` is falsy for lesson_number=0.

        Courses in this system start at Lesson 0. When a search filtered to
        lesson 0 returns nothing, the message should say
        "No relevant content found in lesson 0." but `if 0:` is False in
        Python, so "in lesson 0" is silently dropped.

        Note: VectorStore._build_filter() uses `lesson_number is None` (correct),
        so the actual search is right — only the diagnostic message is broken.

        Fix: change `if lesson_number:` → `if lesson_number is not None:`
        """
        mock_store.search.return_value = make_results(docs=[], metas=[])

        result = tool.execute(query="intro", lesson_number=0)

        assert "in lesson 0" in result, (
            "lesson_number=0 should appear in the no-results message. "
            "`if lesson_number:` treats 0 as falsy. "
            "Fix: use `if lesson_number is not None:` instead."
        )

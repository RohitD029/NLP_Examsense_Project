"""
Streamlit AppTest Suite
=======================
Tests app.py and each page using streamlit.testing.v1.AppTest:
- empty library
- one doc with mocked LLM answer
- mocked 429
- no key
- Teacher mode ON and OFF
Guarantees no exceptions and no crash cards.
"""
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_app_no_key_and_teacher_mode_toggle():
    """Test app entry point with no key, and test Teacher mode toggling on/off."""
    with patch("core.generation.llm_client.get_active_groq_key", return_value=""):
        # 1. Default run (Teacher mode OFF)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        assert not at.exception
        # Confirm no crash card
        for md in at.markdown:
            assert "Unable to display page" not in md.value
        assert at.sidebar is not None

        # 2. Toggle Teacher mode ON
        at_on = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
        at_on.session_state["teacher_mode"] = True
        at_on.run()
        assert not at_on.exception
        assert at_on.session_state["teacher_mode"] is True

        # 3. Toggle Teacher mode OFF
        at_off = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
        at_off.session_state["teacher_mode"] = False
        at_off.run()
        assert not at_off.exception
        assert at_off.session_state["teacher_mode"] is False


def test_library_empty_and_with_document():
    """Test library page when empty and when a document is present."""
    # 1. Empty library
    at = AppTest.from_file(str(ROOT / "pages/library.py"), default_timeout=15).run()
    assert not at.exception
    content = " ".join(m.value for m in at.markdown)
    assert "Your Library is Empty" in content

    # 2. Populate session store inside AppTest's session state
    at_with_doc = AppTest.from_file(str(ROOT / "pages/library.py"), default_timeout=15)
    at_with_doc.session_state["_session_store"] = {
        "documents": {
            "machine_learning.pdf": {
                "doc_name": "machine_learning.pdf",
                "chunks_count": 1,
                "status_note": "Ready",
                "total_pages": 1,
            }
        },
        "chunks": [{"chunk_id": "c1", "text": "Supervised learning.", "metadata": {"doc_name": "machine_learning.pdf", "page_or_slide": 1}}],
        "embeddings": None,
        "faiss_index": None,
        "bm25_data": {},
    }
    at_with_doc.run()
    assert not at_with_doc.exception
    content_doc = " ".join(m.value for m in at_with_doc.markdown)
    assert "machine_learning.pdf" in content_doc
    # Delete button is present
    del_btns = [b for b in at_with_doc.button if b.key == "del_doc_machine_learning.pdf"]
    assert len(del_btns) == 1


def test_ask_page_empty_and_with_mocked_llm():
    """Test ask page when empty, and answering with mocked LLM."""
    # 1. Empty library: shows empty state
    at_empty = AppTest.from_file(str(ROOT / "pages/ask.py"), default_timeout=15).run()
    assert not at_empty.exception
    content_empty = " ".join(m.value for m in at_empty.markdown)
    assert "No Documents Uploaded Yet" in content_empty

    # 2. Add document into AppTest session state
    at_ask = AppTest.from_file(str(ROOT / "pages/ask.py"), default_timeout=15)
    at_ask.session_state["_session_store"] = {
        "documents": {
            "nlp.pdf": {
                "doc_name": "nlp.pdf",
                "chunks_count": 1,
                "status_note": "Ready",
                "total_pages": 1,
            }
        },
        "chunks": [{"chunk_id": "c_nlp", "text": "Transformers rely on attention mechanisms to model token interactions.", "metadata": {"doc_name": "nlp.pdf", "page_or_slide": 1}}],
        "embeddings": None,
        "faiss_index": None,
        "bm25_data": {},
    }

    # Mock answer generator at core module level
    with patch("core.generation.answer_generator.generate_study_answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "Transformers use self-attention to model sequence interactions [1].",
            "source_chips": [
                {
                    "label": "nlp.pdf · p.1",
                    "doc_name": "nlp.pdf",
                    "page_or_slide": 1,
                    "heading": "Intro",
                    "excerpt": "Transformers rely on attention mechanisms.",
                }
            ],
            "not_found": False,
            "error": None,
            "is_summary": False,
        }
        at_ask.run()
        assert not at_ask.exception
        assert len(at_ask.chat_input) > 0
        at_ask.chat_input[0].set_value("How does attention work?").run()
        assert not at_ask.exception

        content_resp = " ".join(m.value for m in at_ask.markdown)
        assert "Transformers use self-attention" in content_resp


def test_ask_page_mocked_429():
    """Test ask page when LLM returns 429 rate limit error."""
    at = AppTest.from_file(str(ROOT / "pages/ask.py"), default_timeout=15)
    at.session_state["_session_store"] = {
        "documents": {
            "nlp.pdf": {
                "doc_name": "nlp.pdf",
                "chunks_count": 1,
                "status_note": "Ready",
                "total_pages": 1,
            }
        },
        "chunks": [{"chunk_id": "c_nlp", "text": "Transformers rely on attention mechanisms.", "metadata": {"doc_name": "nlp.pdf", "page_or_slide": 1}}],
        "embeddings": None,
        "faiss_index": None,
        "bm25_data": {},
    }

    with patch("core.generation.answer_generator.generate_study_answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "",
            "source_chips": [],
            "not_found": False,
            "error": "Rate limit reached on Groq. Please wait a moment or use your own key in the sidebar.",
            "is_summary": False,
        }
        at.run()
        assert not at.exception
        assert len(at.chat_input) > 0
        at.chat_input[0].set_value("Explain attention").run()
        assert not at.exception
        content = " ".join(m.value for m in at.markdown)
        assert "Rate limit reached on Groq" in content
        # No crash card
        assert "Unable to display page" not in content


def test_evaluation_page_teacher_mode():
    """Test evaluation page with teacher mode OFF and ON."""
    # 1. Teacher mode OFF
    at_off = AppTest.from_file(str(ROOT / "pages/evaluation.py"), default_timeout=15).run()
    assert not at_off.exception
    content_off = " ".join(m.value for m in at_off.info)
    assert "Teacher Mode is currently OFF" in content_off

    # 2. Teacher mode ON
    at_on = AppTest.from_file(str(ROOT / "pages/evaluation.py"), default_timeout=15)
    at_on.session_state["teacher_mode"] = True
    at_on.run()
    assert not at_on.exception
    content_on = " ".join(m.value for m in at_on.markdown)
    assert "No Documents Uploaded" in content_on

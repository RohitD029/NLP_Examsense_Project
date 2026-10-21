"""Unit tests for core/nlp/preprocess.py."""
from core.nlp.preprocess import clean_text, bm25_tokenize


def test_clean_text_unicode_normalization():
    # Fullwidth letters or compatibility characters
    text = "Ｔｅｓｔ　ｕｎｉｃｏｄｅ"
    cleaned = clean_text(text)
    assert cleaned == "Test unicode"


def test_clean_text_hyphenated_line_breaks():
    text = "This is a demon-\nstration of hyphen-\n  ation fix."
    cleaned = clean_text(text)
    assert "demonstration" in cleaned
    assert "hyphenation" in cleaned


def test_clean_text_page_number_dropping():
    text = (
        "Introduction to NLP\n"
        "Page 1\n"
        "Here is the content of the first page.\n"
        "12\n"
        "Here is more content.\n"
        "Page 2 of 20\n"
    )
    cleaned = clean_text(text)
    assert "Page 1" not in cleaned
    assert "Page 2 of 20" not in cleaned
    assert "12" not in cleaned.split()
    assert "Introduction to NLP" in cleaned
    assert "first page" in cleaned


def test_clean_text_collapse_whitespace():
    text = "Too   many    spaces   and\n\n\n\nmultiple   blank lines."
    cleaned = clean_text(text)
    assert "Too many spaces and" in cleaned
    assert "\n\n\n" not in cleaned


def test_bm25_tokenize():
    text = "The Transformer models are evaluating, quickly and efficiently!"
    tokens = bm25_tokenize(text)
    # Stopwords like "the", "are", "and" should be removed
    assert "the" not in tokens
    assert "are" not in tokens
    assert "and" not in tokens
    # Punctuation removed, lowercased, and stemmed
    assert any("transform" in t for t in tokens)
    assert any("evaluat" in t for t in tokens)
    assert any("quick" in t for t in tokens)
    assert any("effici" in t for t in tokens)


def test_bm25_tokenize_empty():
    assert bm25_tokenize("") == []
    assert bm25_tokenize("the is at of on") == []

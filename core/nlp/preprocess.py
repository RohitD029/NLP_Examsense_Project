"""
Text Preprocessing
===================
Clean text for chunking and BM25 tokenization.
No external NLTK or spacy download dependencies.
"""
from __future__ import annotations

import re
import unicodedata

# Hardcoded English stopwords (no external download needed)
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}


def _simple_stem(word: str) -> str:
    """Lightweight rule-based stemmer (no download required)."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4 and word[-3] in ("s", "x", "z", "c", "h"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    return word


def clean_text(text: str) -> str:
    """
    1) Normalize unicode (NFKC)
    2) Fix hyphenated line breaks (e.g. "connec-\\ntion" -> "connection")
    3) Drop page numbers (e.g. standalone numbers or "Page X of Y")
    4) Collapse excessive whitespace
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix hyphenated line breaks
    text = re.sub(r"(\w+)-\s*\n+\s*(\w+)", r"\1\2", text)

    # 3. Process line-by-line to drop page numbers and redundant headers
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Drop standalone page numbers like "12", "Page 12", "Page 12 of 30", "- 12 -"
        if re.match(r"^(page\s+\d+(\s+of\s+\d+)?|-?\s*\d+\s*-?)$", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)

    # 4. Collapse multiple blank lines to double newlines and collapse inline whitespace
    processed_lines = []
    for line in cleaned_lines:
        line = re.sub(r"[ \t]+", " ", line)
        processed_lines.append(line)

    result = "\n".join(processed_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def bm25_tokenize(text: str) -> list[str]:
    """
    1) Lowercase
    2) Remove punctuation / extract alphanumeric words
    3) Remove stopwords using hardcoded set
    4) Apply simple stemmer (no download required)
    """
    if not text:
        return []

    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for w in words:
        if len(w) > 1 and w not in _STOPWORDS:
            tokens.append(_simple_stem(w))
    return tokens

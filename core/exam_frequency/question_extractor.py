"""
Question Extractor
==================
Extracts individual questions from past question paper text.
Uses regex and heuristic splitting on common question numbering patterns.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_questions(
    text: str,
    source_document: str = "",
    year: Optional[str] = None,
    subject: Optional[str] = None,
) -> list[dict]:
    """
    Extract individual questions from a past question paper's text.

    Recognizes common patterns:
    - "Q1.", "Q.1", "Q1)", "Q 1."
    - "1.", "1)", "1:"
    - "Question 1", "Question 1:"
    - "(a)", "(b)", etc. as sub-questions
    - Roman numerals: "i)", "ii)", "iii)"

    Args:
        text: Full text of a question paper.
        source_document: Source document name for metadata.
        year: Year tag if available.
        subject: Subject tag if available.

    Returns:
        List of question dicts.
    """
    if not text.strip():
        return []

    # Primary question pattern: matches main question numbers
    main_pattern = re.compile(
        r'(?:^|\n)\s*'
        r'(?:'
        r'[Qq](?:uestion)?\s*\.?\s*(\d+)\s*[.):]\s*'  # Q1. or Question 1:
        r'|(\d{1,2})\s*[.):](?!\d)\s*'                  # 1. or 1) or 1:
        r')',
        re.MULTILINE,
    )

    # Sub-question pattern: (a), (b), (i), (ii) etc.
    sub_pattern = re.compile(
        r'(?:^|\n)\s*'
        r'(?:'
        r'\(([a-z])\)\s*'                # (a), (b), etc.
        r'|([a-z])\)\s*'                 # a), b), etc.
        r'|\(([ivxlc]+)\)\s*'            # (i), (ii), etc.
        r'|([ivxlc]+)\)\s*'              # i), ii), etc.
        r')',
        re.MULTILINE,
    )

    # Find all main question boundaries
    matches = list(main_pattern.finditer(text))

    if not matches:
        return _fallback_extract(text, source_document, year, subject)

    questions = []

    for i, match in enumerate(matches):
        q_num = match.group(1) or match.group(2)
        q_number_label = f"Q{q_num}"

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        q_text = text[start:end].strip()

        if not q_text or len(q_text) < 10:
            continue

        sub_matches = list(sub_pattern.finditer(q_text))

        if sub_matches and len(sub_matches) > 1:
            for j, sub_match in enumerate(sub_matches):
                sub_label = (
                    sub_match.group(1) or sub_match.group(2) or
                    sub_match.group(3) or sub_match.group(4) or "?"
                )
                sub_start = sub_match.end()
                sub_end = (
                    sub_matches[j + 1].start()
                    if j + 1 < len(sub_matches)
                    else len(q_text)
                )
                sub_text = q_text[sub_start:sub_end].strip()

                if sub_text and len(sub_text) >= 10:
                    preamble = q_text[:sub_matches[0].start()].strip()
                    full_sub_text = f"{preamble}\n{sub_text}" if preamble else sub_text

                    questions.append({
                        "question_text": full_sub_text,
                        "question_number": f"{q_number_label}({sub_label})",
                        "source_document": source_document,
                        "year": year,
                        "subject": subject,
                    })
        else:
            questions.append({
                "question_text": q_text,
                "question_number": q_number_label,
                "source_document": source_document,
                "year": year,
                "subject": subject,
            })

    logger.info(
        f"Extracted {len(questions)} questions from '{source_document}' "
        f"(year: {year})"
    )
    return questions


def _fallback_extract(
    text: str,
    source_document: str,
    year: Optional[str],
    subject: Optional[str],
) -> list[dict]:
    """
    Fallback extraction: split by blank lines, keep segments that look
    like questions (contain '?', or start with action verbs).
    """
    question_verbs = {
        "explain", "describe", "define", "compare", "discuss",
        "what", "why", "how", "when", "where", "which", "who",
        "list", "state", "derive", "prove", "solve", "calculate",
        "differentiate", "distinguish", "enumerate", "illustrate",
        "write", "draw", "design", "implement", "analyze", "evaluate",
    }

    paragraphs = re.split(r'\n\s*\n', text)
    questions = []

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 15:
            continue

        first_word = para.split()[0].lower().rstrip(".:,;")
        has_question_mark = "?" in para
        starts_with_verb = first_word in question_verbs

        if has_question_mark or starts_with_verb:
            questions.append({
                "question_text": para,
                "question_number": f"Q{i + 1}",
                "source_document": source_document,
                "year": year,
                "subject": subject,
            })

    logger.info(
        f"Fallback extraction: {len(questions)} questions from '{source_document}'"
    )
    return questions

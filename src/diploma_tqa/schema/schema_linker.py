import re
from difflib import SequenceMatcher


def strip_type_annotation(column: str) -> str:
    """
    Remove trailing angle-bracket annotations if present.
    """
    return re.sub(r"<[^<>]+>$", "", column)


def extract_type_annotation(column: str) -> str | None:
    match = re.search(r"<([^<>]+)>$", column)
    if match:
        return match.group(1)
    return None


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set[str]:
    return set(normalize_text(text).split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def score_column(question: str, column: str) -> float:
    question_norm = normalize_text(question)

    base = strip_type_annotation(column)
    base_norm = normalize_text(base)

    question_tokens = tokenize(question)
    column_tokens = tokenize(base)

    score = 0.0

    # Strong signal: normalized column name appears in the question.
    if base_norm and base_norm in question_norm:
        score += 3.0

    # Medium signal: shared words between question and column.
    score += 1.5 * len(question_tokens & column_tokens)

    # Weak signal: partial token match.
    for col_token in column_tokens:
        for q_token in question_tokens:
            if len(col_token) >= 4 and len(q_token) >= 4:
                if col_token in q_token or q_token in col_token:
                    score += 0.5

    # Weak fallback: fuzzy similarity.
    score += 0.5 * similarity(base_norm, question_norm)

    return score


def find_relevant_columns(
    question: str,
    columns: list[str],
    top_k: int = 8,
    min_score: float = 0.5,
) -> list[str]:
    scored = []

    for column in columns:
        score = score_column(question, column)
        scored.append((score, column))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [column for score, column in scored if score >= min_score][:top_k]


def make_schema_hint(question: str, columns: list[str], top_k: int = 8) -> str:
    relevant_columns = find_relevant_columns(
        question=question,
        columns=columns,
        top_k=top_k,
    )

    if not relevant_columns:
        return "No high-confidence schema links found."

    lines = ["Likely relevant columns:"]

    for column in relevant_columns:
        base = strip_type_annotation(column)
        annotation = extract_type_annotation(column)

        if annotation:
            lines.append(f"- {column}  # base name: {base}, annotation: {annotation}")
        else:
            lines.append(f"- {column}")

    return "\n".join(lines)
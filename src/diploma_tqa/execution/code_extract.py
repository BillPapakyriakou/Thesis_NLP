import re
import textwrap


def strip_code_fences(text: str) -> str:

    # Remove markdown code fences from the model response (extracts code from python blocks)
    text = text.strip()

    if "```" in text:
        match = re.search(
            r"```(?:python)?\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

    return text


def normalize_body_indentation(body: str) -> str:
    """
    Normalize model-generated code before placing it inside:

        def answer(df):
            <body>

    This fixes common LLM outputs like:

        result = ...
            return result

    while preserving normal nested blocks such as if/else bodies.
    """
    body = textwrap.dedent(body).strip()
    lines = body.splitlines()

    normalized = []
    previous_ended_with_colon = False

    for line in lines:
        if not line.strip():
            normalized.append("")
            continue

        stripped = line.lstrip()

        starts_top_level = stripped.startswith(
            (
                "result",
                "return ",
                "#",
            )
        )

        looks_like_assignment = (
            "=" in stripped
            and not stripped.startswith(
                (
                    "if ",
                    "elif ",
                    "else:",
                    "for ",
                    "while ",
                    "try:",
                    "except ",
                    "with ",
                    "return ",
                    "#",
                )
            )
        )

        # If the previous line opened a block, preserve indentation for this line.
        if previous_ended_with_colon:
            normalized.append("    " + stripped)
        elif starts_top_level or looks_like_assignment:
            normalized.append(stripped)
        else:
            normalized.append(stripped)

        previous_ended_with_colon = stripped.endswith(":")

    return "\n".join(normalized).strip()


def extract_answer_body(text: str) -> str:
    """
    Extract code that should run inside:

        def answer(df):
            <body>

    Handles:
    - full def answer(df): blocks
    - code fences
    - explanations after code fences
    - bare code with result/return
    """
    text = strip_code_fences(text)

    # Remove imports; executor provides helper libraries already.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            continue
        if stripped.startswith("from "):
            continue
        lines.append(line)

    text = "\n".join(lines).strip()

    # Case 1: model returned full answer(df) function
    if "def answer" in text:
        lines = text.splitlines()
        body_lines = []
        inside = False

        for line in lines:
            if line.strip().startswith("def answer"):
                inside = True
                continue

            if inside:
                if line.strip() == "":
                    body_lines.append(line)
                    continue

                # Stop when the model response leaves the function body
                if not line.startswith((" ", "\t")):
                    break

                body_lines.append(line)

        body = "\n".join(body_lines).strip()

    # Case 2: model returned bare code
    else:
        body = text.strip()

    # ensure code returns a value
    if "return " not in body:
        body = body + "\nreturn result"

    body = normalize_body_indentation(body)

    if not body:
        raise ValueError("No executable answer body found.")

    # Executor expects the body to be inside def answer(df)
    return textwrap.indent(body, "    ")
import re
import textwrap


def strip_code_fences(text: str) -> str:
    text = text.strip()

    if "```" in text:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return text


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

    # Remove imports; executor provides pd/np already.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            continue
        if stripped.startswith("from "):
            continue
        lines.append(line)

    text = "\n".join(lines).strip()

    # Case 1: model returned full function
    if "def answer" in text:
        lines = text.splitlines()
        body_lines = []
        inside = False

        for line in lines:
            if line.strip().startswith("def answer"):
                inside = True
                continue

            if inside:
                # Keep blank lines
                if line.strip() == "":
                    body_lines.append(line)
                    continue

                # Stop at next top-level code block
                if not line.startswith((" ", "\t")):
                    break

                body_lines.append(line)

        body = textwrap.dedent("\n".join(body_lines)).strip()

    # Case 2: model returned bare code
    else:
        body = text.strip()

    # If body has no return statement, assume it created `result`.
    if "return " not in body:
        body = body + "\nreturn result"

        # Normalize indentation before adding our own indentation.
        # This prevents cases like:
        #     result = ...
        #         return result
    body = textwrap.dedent(body).strip()

    if not body:
        raise ValueError("No executable answer body found.")

    return textwrap.indent(body, "    ")
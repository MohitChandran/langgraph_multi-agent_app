"""Numbered citations ([1], [2]...) shared by the agents that answer from numbered sources (Doc and Web).

Both agents give the answer LLM a numbered list of sources and ask for "[n] after every sentence that uses
source n". The model can still cite a number that doesn't exist, or forget to list one it used in the text;
clean_citations fixes both, so every [n] the user sees leads to a real source.
"""

import re

CITATION_MARK = re.compile(r"\[(\d+)\]")


def clean_citations(text: str, cited: list[int], source_count: int) -> tuple[str, list[int]]:
    """Keep only citation numbers that are real sources; returns (text without invalid [n], valid numbers in order).

    The numbers come from BOTH the model's `citations` list and the [n] marks in its text (models
    sometimes forget one of the two). An invalid [7] with 6 sources is removed from the text.
    """
    valid = range(1, source_count + 1)
    in_text = [int(n) for n in CITATION_MARK.findall(text)]
    numbers = list(dict.fromkeys(n for n in [*in_text, *cited] if n in valid))  # unique, first-seen order
    cleaned = CITATION_MARK.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
    return cleaned.strip(), numbers

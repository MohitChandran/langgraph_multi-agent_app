"""Prompt templates of the Doc QA agent (plan 02, sections 5.5 and 6.3).

Kept in one file, like the DB agent's prompts, so they can be tuned (and compared in the R9
eval) without touching any logic. R3 adds the vision prompt; R7 adds query-rewrite and answer.
"""

# Describing a figure so it can be FOUND by search and USED to answer questions. The rules
# matter for correctness later: exact values only when printed; estimates marked "≈ ... (estimated
# from axis)" so the answer step can say "approximately"; "unreadable" instead of a guess.
# The last two lines were added in R3: without them the model spent its 250-token budget on
# markdown, a preamble and bar colours, and long diagrams were cut off before their last steps.
VISION_PROMPT = """You are describing a figure from a business document so it can be found by search and used to answer questions.
Context: section "{heading_path}", caption "{caption}".
1) Say what kind of figure it is (bar chart, line chart, pie chart, flow diagram, org chart, table image, screenshot, photo).
2) Transcribe all readable text: title, axis labels, legend, labels on boxes.
3) Charts: list the data as "label: value" lines. Report only values that are printed on the figure or clearly readable.
   If a value must be estimated (e.g. a bar with no printed label, read against the axis), write "≈ value (estimated from
   axis)". Never guess; write "unreadable" instead. Diagrams: list the steps or boxes in order, with what connects to what.
4) Only state what is visible. At most 180 words.
Write plain text lines: no markdown (no ** or bullet symbols), no introduction, and do not describe colours or styling.
Start directly with the kind of figure."""  # noqa: E501 (prompt text: the plan's wording plus the last two lines)

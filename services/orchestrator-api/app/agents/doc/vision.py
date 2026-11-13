"""Pictures: decide which ones to describe, and describe them with a Groq vision model (plan 02, section 5.5).

Why describe pictures at all (decision D5): a chart's numbers or a diagram's steps exist only as
pixels. Search can't find pixels and the answer LLM never sees them. A vision model writes them
down as text once, at upload time, and that text becomes a normal searchable "picture" chunk.

Why filter first: every description is one Groq call, and the free tier allows roughly 4 per
minute for this model, so logos, icons, tiny images and decorative strips are skipped, and at
most RAG_MAX_PICTURES_PER_DOC pictures (charts and diagrams first) are described.

CLI:
    uv run python -m app.agents.doc.vision <image.png> [--caption "..."] [--heading "..."]
    uv run python -m app.agents.doc.vision <document.pdf>      # parse, filter (with reasons), describe
"""

import argparse
import asyncio
import base64
import io
import logging
import time
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image

from app.agents.doc.models import Element
from app.agents.doc.prompts import VISION_PROMPT
from app.config import get_settings
from app.llm import get_llm

logger = logging.getLogger(__name__)

SKIP_LABELS = {"logo", "signature", "stamp", "icon", "qr_code", "bar_code"}
PRIORITY_LABELS = {"bar_chart", "line_chart", "pie_chart", "flow_chart", "table", "screenshot", "map"}
MIN_SIDE_PX = 150
MAX_ASPECT_RATIO = 8  # wider/taller than 8:1 without a caption = decorative strip
VISION_MAX_TOKENS = 250  # ~180 words; also keeps us under Groq's output-tokens-per-minute limit
MAX_IMAGE_SIDE_PX = 1600  # Groq limits the request size; bigger images add nothing for a chart


def skip_reason(element: Element, min_area: float) -> str | None:
    """Why this picture should NOT be described, or None if it should.

    Checked in order: classifier label, pixel size, share of the page, decorative strip shape.
    A picture without a saved image can't be sent to the vision model at all.
    """
    if element.image_path is None:
        return "no image could be extracted"
    if element.picture_label in SKIP_LABELS:
        return f"label '{element.picture_label}'"
    if (element.width_px or 0) < MIN_SIDE_PX or (element.height_px or 0) < MIN_SIDE_PX:
        return f"too small ({element.width_px}x{element.height_px} px, minimum {MIN_SIDE_PX})"
    if element.area_fraction is not None and element.area_fraction < min_area:
        return f"covers {element.area_fraction:.1%} of the page (minimum {min_area:.0%})"
    long_side = max(element.width_px, element.height_px)
    short_side = min(element.width_px, element.height_px)
    if long_side / short_side > MAX_ASPECT_RATIO and not element.text:
        return f"decorative strip ({long_side / short_side:.0f}:1, no caption)"
    return None


def select_pictures(
    elements: list[Element], max_n: int, min_area: float
) -> tuple[list[Element], list[tuple[Element, str]]]:
    """Split a document's pictures into (to_describe, skipped-with-reason).

    The cap keeps the most informative pictures: charts, diagrams, screenshots and maps
    first, then everything else, each group in page order. to_describe is returned in
    document order (so progress "pictures 3/12" follows the pages).
    """
    pictures = [element for element in elements if element.kind == "picture"]
    skipped: list[tuple[Element, str]] = []
    candidates: list[Element] = []
    for picture in pictures:
        reason = skip_reason(picture, min_area)
        if reason:
            skipped.append((picture, reason))
        else:
            candidates.append(picture)

    ranked = sorted(candidates, key=lambda p: (p.picture_label not in PRIORITY_LABELS, p.page_start))
    kept_ids = {id(picture) for picture in ranked[:max_n]}
    skipped.extend((picture, f"over the cap of {max_n} pictures per document") for picture in ranked[max_n:])
    return [picture for picture in candidates if id(picture) in kept_ids], skipped


def image_data_url(path: str) -> str:
    """Load an image, shrink it if a side is over MAX_IMAGE_SIDE_PX, and return it as a base64 PNG data URL."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX))  # keeps the aspect ratio; never enlarges
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


async def describe_picture(element: Element, heading_path: str, llm: BaseChatModel | None = None) -> str | None:
    """Ask the vision model for a factual description of one picture. None on failure (never raises).

    One failed picture must never fail the whole document (plan gotcha 13): the caller keeps
    the caption-only chunk and moves on. The Groq client has already retried rate limits and
    server errors by the time an exception reaches us.
    """
    llm = llm or get_llm("vision", max_tokens=VISION_MAX_TOKENS)
    prompt = VISION_PROMPT.format(heading_path=heading_path or "(none)", caption=element.text or "(none)")
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(element.image_path)}},
        ]
    )
    try:
        reply = await llm.ainvoke([message])
    except Exception:
        logger.exception("Vision description failed for %s", element.image_path)
        return None
    text = str(reply.content).strip()
    return text or None


# ---------------------------------------------------------------- CLI


async def _describe_image_file(path: Path, caption: str, heading: str) -> None:
    """CLI mode 1: describe a single image file."""
    with Image.open(path) as image:
        width, height = image.size
    element = Element("picture", caption, 1, 1, image_path=str(path), width_px=width, height_px=height)
    started = time.perf_counter()
    print(await describe_picture(element, heading))
    print(f"\n[{time.perf_counter() - started:.1f}s]")


async def _describe_document(path: Path) -> None:
    """CLI mode 2: parse a document, show which pictures are skipped and why, and describe the rest."""
    from app.agents.doc.chunking import heading_paths  # local imports: only this CLI mode needs Docling
    from app.agents.doc.parsing import build_converter, parse_document

    settings = get_settings()
    parsed = await asyncio.to_thread(parse_document, path, path.parent / f"{path.stem}-pictures", build_converter())
    paths = dict(zip(map(id, parsed.elements), heading_paths(parsed.elements), strict=True))
    to_describe, skipped = select_pictures(
        parsed.elements, settings.rag_max_pictures_per_doc, settings.rag_min_picture_area
    )
    for picture, reason in skipped:
        print(f"SKIP  p{picture.page_start} label={picture.picture_label}: {reason}")
    for number, picture in enumerate(to_describe, start=1):
        started = time.perf_counter()
        picture.description = await describe_picture(picture, paths[id(picture)])
        print(f"\n=== picture {number}/{len(to_describe)}  p{picture.page_start} label={picture.picture_label} "
              f"caption={picture.text!r}  [{time.perf_counter() - started:.1f}s]\n{picture.description}")  # fmt: skip


def main() -> None:
    """CLI entry point: an image file is described directly; any other file is parsed first."""
    parser = argparse.ArgumentParser(description="Describe pictures with the Groq vision model.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--caption", default="")
    parser.add_argument("--heading", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        asyncio.run(_describe_image_file(args.file, args.caption, args.heading))
    else:
        asyncio.run(_describe_document(args.file))


if __name__ == "__main__":
    main()

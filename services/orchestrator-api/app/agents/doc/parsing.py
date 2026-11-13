"""Parse a document with Docling into our own types (plan 02, section 5.4).

Why Docling (decision D1): a PDF is just positioned glyphs and images. Docling rebuilds what a
human sees: reading order, headings and their levels, real table cells, picture crops with a
label (bar chart, logo, ...), and it runs OCR on scanned pages. It runs on the CPU here; it's
slow-ish (seconds per page), but it only runs once per upload, in the background.

This module is the ONLY place that touches Docling objects. It returns a ParsedDocument
(models.py), so the rest of the pipeline doesn't care which parser produced it.

CLI:
    uv run python -m app.agents.doc.parsing <file>                       # structure only
    uv run python -m app.agents.doc.parsing <file> --chunks              # + pictures described + chunks
    uv run python -m app.agents.doc.parsing <file> --chunks --no-vision  # chunks without Groq calls
"""

import argparse
import asyncio
import logging
import os
import re
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from app.agents.doc.models import Element, ParsedDocument

# Text labels we keep as body text. `caption` is NOT here on purpose: captions are attached to
# their table/picture via caption_text(); keeping them here too would duplicate them (gotcha 10).
# "1 Introduction", "4.2 SLA Terms", "A.1 ...": a numbered section heading, never the document title.
NUMBERED_HEADING = re.compile(r"^(\d+|[A-Z])(\.\d+)*[.)]?\s")
# A paragraph that is really a table caption ("Table 2: Vendor services ...").
TABLE_CAPTION = re.compile(r"^Table\s+\w+[:.]", re.IGNORECASE)

TEXT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.CODE,
}


def build_converter(num_threads: int | None = None) -> DocumentConverter:
    """Create the Docling converter. Expensive (loads layout, table and OCR models): build it ONCE and reuse it.

    PDF options, each one deliberate:
      - do_ocr: read scanned pages / text inside images (RapidOCR in the standard install).
      - TableFormer ACCURATE + cell matching: real rows and columns, not a blob of text.
      - generate_picture_images + images_scale=2: save sharp crops for the vision model.
      - do_picture_classification: label pictures (bar_chart, logo, ...) so we can skip junk.
      - do_picture_description=False: we describe pictures ourselves with Groq (vision.py).
      - heading_hierarchy_options.enabled: infer heading LEVELS from bookmarks, then numbering
        ("4" > "4.2"), then font size. It's OFF by default, and then every PDF heading is level 1,
        so "4.2 SLA Terms" could not nest under "4 Cloud Vendors" (plan gotcha 14).
    DOCX, PPTX, Markdown and images use Docling's defaults: their structure is read directly.
    """
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = True
    options.generate_picture_images = True
    options.images_scale = 2.0
    options.do_picture_classification = True
    options.do_picture_description = False
    options.heading_hierarchy_options.enabled = True
    threads = num_threads or min(8, os.cpu_count() or 4)
    options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=threads)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def parse_document(path: Path, pictures_dir: Path, converter: DocumentConverter) -> ParsedDocument:
    """Convert one file and adapt the result into a ParsedDocument; picture crops are saved in pictures_dir.

    Blocking and CPU-heavy: callers in async code must run it in asyncio.to_thread (gotcha 7).
    """
    document = converter.convert(path).document
    return to_parsed_document(document, fallback_title=path.stem, pictures_dir=pictures_dir)


def to_parsed_document(document: DoclingDocument, fallback_title: str, pictures_dir: Path) -> ParsedDocument:
    """Walk Docling's items in reading order and turn each one into an Element (the adapter rules of 5.4).

    iterate_items() covers only the BODY layer by default, so page headers, footers ("Page 3")
    and watermarks are already left out.
    """
    pictures_dir.mkdir(parents=True, exist_ok=True)
    title: str | None = None
    elements: list[Element] = []
    pictures_per_page: dict[int, int] = {}

    for item, _depth in document.iterate_items():
        page_start, page_end = _pages(item)
        # Order matters: TitleItem and SectionHeaderItem are subclasses of TextItem.
        if isinstance(item, TitleItem):
            title = title or item.text.strip()  # the title is NOT part of any heading path
        elif isinstance(item, SectionHeaderItem):
            if item.text.strip():
                elements.append(Element("heading", item.text.strip(), page_start, page_end, level=item.level))
        elif isinstance(item, TableItem):
            elements.append(_table_element(item, document, page_start, page_end))
        elif isinstance(item, PictureItem):
            pictures_per_page[page_start] = pictures_per_page.get(page_start, 0) + 1
            image_path = pictures_dir / f"p{page_start:03d}-{pictures_per_page[page_start]:02d}.png"
            elements.append(_picture_element(item, document, page_start, page_end, image_path))
        elif isinstance(item, TextItem) and item.label in TEXT_LABELS and item.text.strip():
            elements.append(Element("text", item.text.strip(), page_start, page_end))

    if title is None:
        title = _promote_title(elements)
    return ParsedDocument(
        title=title or fallback_title,
        page_count=max(1, len(document.pages)),
        elements=_attach_table_captions(elements),
    )


def _promote_title(elements: list[Element]) -> str | None:
    """When Docling found no TitleItem, use the document's first heading as the title, if it looks like one.

    Docling often labels a report's big title as a section heading. It's taken as the title only
    if it is on page 1, comes before any body text, and is NOT numbered ("1 Introduction" is a
    section). It's removed from the elements, so it doesn't become part of every heading path.
    """
    for index, element in enumerate(elements):
        if element.kind == "text" or element.page_start > 1:
            return None
        if element.kind == "heading":
            if NUMBERED_HEADING.match(element.text):
                return None
            del elements[index]
            return element.text
    return None


def _attach_table_captions(elements: list[Element]) -> list[Element]:
    """Give a caption-less table the "Table N: ..." paragraph right before it, and drop that paragraph.

    Docling doesn't always link a caption to its table. A table chunk that carries its caption
    ("Table 2: Vendor services, contract owners ...") is far easier to find.
    """
    result: list[Element] = []
    for element in elements:
        previous = result[-1] if result else None
        if (
            element.kind == "table"
            and not element.text
            and previous is not None
            and previous.kind == "text"
            and TABLE_CAPTION.match(previous.text)
        ):
            element.text = result.pop().text
        result.append(element)
    return result


def _pages(item: object) -> tuple[int, int]:
    """First and last page (1-based) of an item. Formats without pages (DOCX, Markdown) count as page 1."""
    provenance = getattr(item, "prov", None)
    if not provenance:
        return 1, 1
    return provenance[0].page_no, provenance[-1].page_no


def _table_element(item: TableItem, document: DoclingDocument, page_start: int, page_end: int) -> Element:
    """A table as rows of cell strings, header row first. The caption becomes the element's text.

    If Docling found no header row, the DataFrame's column names are just 0, 1, 2...; we keep
    them as-is here and the chunker decides to use the first data row as the header instead.
    """
    frame = item.export_to_dataframe(document)
    header = [str(column) for column in frame.columns]
    rows = frame.astype(str).values.tolist()
    return Element("table", item.caption_text(document).strip(), page_start, page_end, table_rows=[header, *rows])


def _picture_element(
    item: PictureItem, document: DoclingDocument, page_start: int, page_end: int, image_path: Path
) -> Element:
    """A picture: save its crop as PNG and record what the picture filter needs (label, size, page area)."""
    element = Element("picture", item.caption_text(document).strip(), page_start, page_end)

    image = item.get_image(document)
    if image is not None:
        image.save(image_path)
        element.image_path = str(image_path)
        element.width_px, element.height_px = image.size

    if item.meta is not None and item.meta.classification is not None:
        element.picture_label = item.meta.classification.get_main_prediction().class_name

    if item.prov and item.prov[0].page_no in document.pages:
        page = document.pages[item.prov[0].page_no]
        page_area = page.size.width * page.size.height
        element.area_fraction = item.prov[0].bbox.area() / page_area if page_area else None
    return element


# ---------------------------------------------------------------- CLI


def _print_parsed(parsed: ParsedDocument) -> None:
    """Print the document's structure: headings as a tree, then one line per text/table/picture."""
    print(f"Title: {parsed.title}   Pages: {parsed.page_count}\n")
    for element in parsed.elements:
        where = f"p{element.page_start}" + (f"-{element.page_end}" if element.page_end != element.page_start else "")
        if element.kind == "heading":
            print(f"{'  ' * (element.level - 1)}# [L{element.level}] {element.text}   ({where})")
        elif element.kind == "text":
            print(f"      text  {where:<5} {element.text[:90]!r}")
        elif element.kind == "table":
            rows = element.table_rows or [[]]
            shape = f"{len(rows) - 1} rows x {len(rows[0])} cols"
            print(f"      TABLE {where:<5} {shape}  header={rows[0]}  caption={element.text!r}")
        else:
            area = f"{element.area_fraction:.1%}" if element.area_fraction is not None else "n/a"
            size = f"{element.width_px}x{element.height_px}px" if element.width_px else "no image"
            print(f"      PICT  {where:<5} label={element.picture_label} area={area} {size}  caption={element.text!r}")
    kinds = [element.kind for element in parsed.elements]
    print(f"\n{kinds.count('heading')} headings, {kinds.count('text')} text, {kinds.count('table')} tables, "
          f"{kinds.count('picture')} pictures")  # fmt: skip


async def _print_chunks(parsed: ParsedDocument, use_vision: bool) -> None:
    """Run the rest of the ingestion chain on a parsed document (pictures -> chunks) and print every chunk."""
    from app.agents.doc.chunking import TokenCounter, chunk_document, heading_paths
    from app.agents.doc.vision import describe_picture, select_pictures
    from app.config import get_settings

    settings = get_settings()
    to_describe, skipped = select_pictures(
        parsed.elements, settings.rag_max_pictures_per_doc, settings.rag_min_picture_area
    )
    print(f"\nPictures: {len(to_describe)} to describe, {len(skipped)} skipped "
          f"({'; '.join(reason for _, reason in skipped) or '-'})")  # fmt: skip
    if use_vision:
        paths = dict(zip(map(id, parsed.elements), heading_paths(parsed.elements), strict=True))
        for number, picture in enumerate(to_describe, start=1):
            print(f"  describing picture {number}/{len(to_describe)} (page {picture.page_start})...")
            picture.description = await describe_picture(picture, paths[id(picture)])

    counter = TokenCounter(settings.rag_embedding_model)
    chunks = chunk_document(parsed, counter, settings.rag_chunk_tokens)
    print(f"\n{len(chunks)} chunks (budget {settings.rag_chunk_tokens} tokens of content):\n")
    for chunk in chunks:
        pages = f"p{chunk.page_start}" + (f"-{chunk.page_end}" if chunk.page_end != chunk.page_start else "")
        sizes = f"content={counter.count(chunk.content):>3} tok  embed={chunk.token_count:>3} tok"
        print(f"#{chunk.chunk_index:<3} {chunk.chunk_type:<7} {pages:<5} {sizes}  path={chunk.heading_path!r}")
        lines = chunk.content.splitlines()
        for line in lines[:4]:
            print(f"        {line[:110]}")
        if len(lines) > 4:
            print(f"        ... {len(lines) - 4} more lines")


def main() -> None:
    """CLI: parse one file and print its structure; with --chunks, also describe pictures and print the chunks."""
    parser = argparse.ArgumentParser(description="Parse a document with Docling and show its structure.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--pictures-dir", type=Path, help="where to save picture crops (default: next to the file)")
    parser.add_argument("--chunks", action="store_true", help="also describe pictures and print the chunks")
    parser.add_argument("--no-vision", action="store_true", help="with --chunks: skip the Groq picture descriptions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    pictures_dir = args.pictures_dir or args.file.parent / f"{args.file.stem}-pictures"
    started = time.perf_counter()
    converter = build_converter()
    loaded = time.perf_counter()
    parsed = parse_document(args.file, pictures_dir, converter)
    done = time.perf_counter()

    _print_parsed(parsed)
    print(f"Models loaded in {loaded - started:.1f}s; parsed in {done - loaded:.1f}s "
          f"({(done - loaded) / parsed.page_count:.1f}s per page). Pictures saved to {pictures_dir}")  # fmt: skip
    if args.chunks:
        asyncio.run(_print_chunks(parsed, use_vision=not args.no_vision))


if __name__ == "__main__":
    main()

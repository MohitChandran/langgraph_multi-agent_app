# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab==5.1.0", "matplotlib==3.11.2", "pillow==13.0.0", "numpy==2.5.4"]
# ///
"""Generate two 7-page test documents for trying out the Doc QA (RAG) agent by hand.

Run from the repo root:
    uv run evals/make_test_documents.py
Output (upload them in the chat UI with the paperclip):
    evals/test_documents/wensley-parish-registers-guide.pdf      (doc 1: a historical guide)
    evals/test_documents/register-digitisation-report-2026.pdf   (doc 2: a project report)
Questions with expected answers: evals/datasets/test_documents_questions.json

Both documents contain the scanned preface image in assets/ (a real page, low resolution), plus
paragraphs, tables and charts, so every ingestion path is exercised:
  - scanned image with text      -> OCR and/or the vision model must read it
  - tables                       -> table chunks with headers
  - charts WITH printed values   -> exact numbers
  - charts WITHOUT printed values-> answers must say "approximately"
  - pie charts and a flow diagram
Everything except the scanned preface is FICTIONAL (every page says so in its footer): the
counts, budgets and names are illustrative test data, not historical records.

Deterministic (fixed data, reportlab invariant mode, no image timestamps), like scripts/make_sample_docs.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageFilter  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "test_documents"
ASSETS = OUT_DIR / "assets"
PREFACE_ORIGINAL = ASSETS / "wensley-preface-original.png"
FOOTER = "Fictional test document for the Ops Assistant RAG evaluation. Figures are illustrative."

STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=STYLES["BodyText"], fontSize=10.5, leading=14.5, spaceAfter=7)
H1 = ParagraphStyle("H1", parent=STYLES["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=STYLES["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=9, textColor=colors.HexColor("#444444"), spaceAfter=10)
TITLE = ParagraphStyle("DocTitle", parent=STYLES["Title"], fontSize=21, leading=26, spaceAfter=4)

# ---------------------------------------------------------------- doc 1 data
DECADES = [f"{year}s" for year in range(1700, 1840, 10)]
BAPTISMS = [212, 198, 225, 240, 231, 256, 268, 281, 297, 305, 318, 296, 287, 274]
MARRIAGES = [48, 52, 55, 61, 58, 64, 67, 70, 72, 75, 79, 74, 70, 68]
BURIALS = [185, 201, 178, 190, 222, 205, 198, 214, 236, 219, 228, 241, 209, 196]
OCCUPATIONS = {"Farmer": 142, "Labourer": 118, "Lead miner": 64, "Blacksmith": 21, "Weaver": 17, "Innkeeper": 9}
CONDITION_SHARES = {"Good": 61, "Fair": 27, "Poor": 9, "Missing": 3}

# ---------------------------------------------------------------- doc 2 data
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PAGES_PER_MONTH = [120, 145, 180, 210, 235, 260, 190, 150, 230, 245, 220, 155]  # sums to 2,340
OCR_ACCURACY = {
    "Typeset print": 98, "Printed forms": 95, "Copperplate hand": 84, "Secretary hand": 62, "Damaged pages": 55,
}  # fmt: skip
SPEND = {"Staff": 88_400, "Equipment": 29_750, "Conservation": 21_300, "Software & hosting": 6_750, "Training": 2_000}


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    """Shorthand for a paragraph."""
    return Paragraph(text, style)


def bullets(items: list[str]) -> ListFlowable:
    """A bulleted list."""
    return ListFlowable([ListItem(p(item), leftIndent=12) for item in items], bulletType="bullet")


def image(path: Path, width: float) -> Image:
    """An image flowable `width` wide whose height keeps the picture's real aspect ratio (no stretching)."""
    with PILImage.open(path) as picture:
        ratio = picture.height / picture.width
    return Image(str(path), width, width * ratio)


def table(rows: list[list[str]], widths: list[float], total_row: bool = False) -> Table:
    """A grid table with a shaded, bold header row (and optionally a bold last row for totals)."""
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if total_row:
        style.append(("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 9))
    result = Table(rows, colWidths=widths)
    result.setStyle(TableStyle(style))
    return result


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG without timestamps (deterministic) and close it."""
    path = ASSETS / name
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def prepare_preface_scan() -> Path:
    """Upscale the low-resolution scan 2.5x and sharpen it a little: same content, bigger letters for OCR."""
    with PILImage.open(PREFACE_ORIGINAL) as image:
        bigger = image.convert("RGB").resize((image.width * 5 // 2, image.height * 5 // 2), PILImage.Resampling.LANCZOS)
    path = ASSETS / "wensley-preface-2.5x.png"
    bigger.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2)).save(path)
    return path


# ---------------------------------------------------------------- charts


def entries_chart() -> Path:
    """Doc 1, line chart WITHOUT printed values: entries per decade (values must be read from the axis)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(DECADES, BAPTISMS, marker="o", label="Baptisms", color="#4c78a8")
    ax.plot(DECADES, BURIALS, marker="s", label="Burials", color="#e45756")
    ax.plot(DECADES, MARRIAGES, marker="^", label="Marriages", color="#54a24b")
    ax.set_title("Register entries per decade, 1700s-1830s")
    ax.set_ylabel("Entries per decade")
    ax.set_yticks(range(0, 351, 50))
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.legend(loc="center right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "doc1-entries-per-decade.png")


def occupations_chart() -> Path:
    """Doc 1, bar chart WITH printed values: fathers' occupations in baptism entries."""
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bars = ax.bar(list(OCCUPATIONS), list(OCCUPATIONS.values()), color="#9ecae9", edgecolor="#4c78a8")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Fathers' occupations in baptism entries, 1813-1837")
    ax.set_ylabel("Baptism entries")
    ax.set_ylim(0, 165)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "doc1-occupations.png")


def pie_with_legend(
    values: dict[str, float], title: str, colours: list[str] | None = None, decimals: int = 0
) -> plt.Figure:
    """A pie whose small slices don't get overlapping labels: percentages are printed only inside big
    slices, and EVERY category with its percentage is listed in the legend (readable for the vision model)."""
    total = sum(values.values())
    shares = {name: value / total * 100 for name, value in values.items()}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    wedges, _texts, _autotexts = ax.pie(
        list(values.values()), colors=colours, startangle=90, counterclock=False,
        autopct=lambda pct: f"{pct:.{decimals}f}%" if pct >= 8 else "", textprops={"fontsize": 9},
    )  # fmt: skip
    ax.legend(wedges, [f"{name}: {share:.{decimals}f}%" for name, share in shares.items()], loc="center left",
              bbox_to_anchor=(1, 0.5), fontsize=9, frameon=False)  # fmt: skip
    ax.set_title(title, fontsize=10)
    return fig


def condition_pie() -> Path:
    """Doc 1, pie chart with percentages: condition of the surveyed pages."""
    colours = ["#54a24b", "#eeca3b", "#e45756", "#bab0ac"]
    return save(pie_with_legend(CONDITION_SHARES, "Condition of surveyed pages (2024)", colours), "doc1-condition.png")


def monthly_chart() -> Path:
    """Doc 2, line chart WITH printed values: pages digitised per month."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(MONTHS, PAGES_PER_MONTH, marker="o", color="#4c78a8")
    for month, value in zip(MONTHS, PAGES_PER_MONTH, strict=True):
        ax.annotate(str(value), (month, value), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_title("Pages digitised per month, 2026")
    ax.set_ylabel("Pages")
    ax.set_ylim(0, 300)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "doc2-pages-per-month.png")


def accuracy_chart() -> Path:
    """Doc 2, bar chart WITHOUT printed values: OCR word accuracy by script type (read from the axis)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh(list(OCR_ACCURACY), list(OCR_ACCURACY.values()), color="#f58518")
    ax.set_title("OCR word accuracy by script type, 2026")
    ax.set_xlabel("Word accuracy (%)")
    ax.set_xticks(range(0, 101, 10))
    ax.grid(axis="x", linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "doc2-ocr-accuracy.png")


def spend_pie() -> Path:
    """Doc 2, pie chart with percentages: actual spend by category."""
    return save(pie_with_legend(SPEND, "Actual spend by category, 2026", decimals=1), "doc2-spend.png")


def workflow_diagram() -> Path:
    """Doc 2, flow diagram: the digitisation workflow, with one decision."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(-0.2, 10.4), ax.set_ylim(0, 5), ax.axis("off")
    boxes = {
        "retrieve": (0.0, 3.6, "Retrieve volume\nfrom store"),
        "check": (2.6, 3.6, "Condition\ncheck"),
        "repair": (5.2, 3.6, "Needs\nrepair?"),
        "conserve": (7.9, 3.6, "Conservation\n(conservator)"),
        "capture": (5.2, 1.8, "Image capture\n(400 dpi)"),
        "qa": (7.9, 1.8, "Quality check\n(10% sample)"),
        "ocr": (7.9, 0.1, "OCR + metadata"),
        "publish": (5.2, 0.1, "Publish online"),
    }
    for key, (x, y, label) in boxes.items():
        color = "#fde2c8" if key == "repair" else "#dbe9f6"
        ax.add_patch(FancyBboxPatch((x, y), 2.1, 0.9, boxstyle="round,pad=0.1", fc=color, ec="#333333"))
        ax.text(x + 1.05, y + 0.45, label, ha="center", va="center", fontsize=8.5)

    def arrow(start: tuple[float, float], end: tuple[float, float], label: str = "", at: tuple = (0, 0)) -> None:
        """Draw an arrow, with an optional label at `at`."""
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#333333"))
        if label:
            ax.text(*at, label, fontsize=9, color="#b03a2e", fontweight="bold")

    arrow((2.2, 4.05), (2.6, 4.05))
    arrow((4.8, 4.05), (5.2, 4.05))
    arrow((7.4, 4.05), (7.9, 4.05), "yes", at=(7.42, 4.62))
    arrow((6.25, 3.6), (6.25, 2.8), "no", at=(6.4, 3.1))
    arrow((8.95, 3.6), (7.4, 2.75))
    arrow((7.4, 2.25), (7.9, 2.25))
    arrow((8.95, 1.8), (8.95, 1.1))
    arrow((7.9, 0.55), (7.4, 0.55))
    ax.set_title("Digitisation workflow", fontsize=11)
    return save(fig, "doc2-workflow.png")


# ---------------------------------------------------------------- document template


class SevenPageDocument(BaseDocTemplate):
    """A4 document: footer with the fiction notice and page number, PDF bookmarks for headings."""

    def __init__(self, filename: Path, title: str) -> None:
        """Margins, deterministic output (invariant=1) and one page template."""
        super().__init__(str(filename), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                         bottomMargin=2 * cm, invariant=1, title=title, author="Ops Assistant test data")  # fmt: skip
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._footer)])
        self._bookmarks = 0

    @staticmethod
    def _footer(canvas, doc) -> None:  # noqa: ANN001 (reportlab callback signature)
        """Fiction notice on the left, page number on the right."""
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(2 * cm, 1.2 * cm, FOOTER)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page} of 7")

    def afterFlowable(self, flowable) -> None:  # noqa: ANN001, N802 (reportlab hook name)
        """Add a PDF bookmark for every H1 (level 0) and H2 (level 1) heading, for Docling's heading levels."""
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            key = f"h{self._bookmarks}"
            self._bookmarks += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0 if flowable.style.name == "H1" else 1)


# ---------------------------------------------------------------- doc 1


def guide_story(scan: Path) -> list:
    """Doc 1: The Wensley Parish Registers 1538-1837, a guide to the printed volume. One PageBreak per page."""
    last_four = zip(DECADES[-4:], BAPTISMS[-4:], MARRIAGES[-4:], BURIALS[-4:], strict=True)
    decades_table = [["Decade", "Baptisms", "Marriages", "Burials"]]
    decades_table += [[decade, str(b), str(m), str(u)] for decade, b, m, u in last_four]
    return [
        # ---- page 1
        p("The Wensley Parish Registers, 1538-1837", TITLE),
        p("A guide to the printed volume, its register books and their condition", CAPTION),
        p("1 Introduction", H1),
        p(
            "The parish registers of Wensley record the baptisms, marriages and burials of the parish for three "
            "centuries. This guide explains which register books the printed volume was transcribed from, how the "
            "number of entries changed over time, who appears in them, and in what condition the original books "
            "survive today."
        ),
        p(
            "The printed volume continues an earlier publication of the same registers. Readers who need entries "
            "before 1701 should use that earlier volume; this guide concentrates on the later register books."
        ),
        p("2 How to Use This Guide", H1),
        p("Each section can be read on its own:"),
        bullets(
            [
                "Section 3 reproduces the preface of the printed volume as a scanned page.",
                "Section 4 lists the six register books and the years they cover.",
                "Sections 5 and 6 describe the entries: their number per decade and the families recorded.",
                "Section 7 summarises the 2024 condition survey; section 8 explains how to see the originals.",
            ]
        ),  # fmt: skip
        p("All counts in sections 5 to 7 are illustrative figures prepared for this guide."),
        PageBreak(),
        # ---- page 2
        p("3 The Society's Preface", H1),
        p(
            "The preface of the printed volume, reproduced below exactly as scanned, explains how the volume relates "
            "to the society's earlier publication and describes the register books."
        ),
        image(scan, 15 * cm),
        p("Figure 1: Scanned preface of the printed register volume", CAPTION),
        PageBreak(),
        # ---- page 3
        p("4 The Register Books", H1),
        p(
            "Six register books survive. The two oldest are parchment books bound in leather; the later ones use the "
            "printed forms introduced for marriages in 1754 and for baptisms and burials in 1813."
        ),
        p("Table 1: Register books used for the printed volume", CAPTION),
        table(
            [
                ["Book", "Material", "Size", "Entries", "Years", "Pages"],
                ["No. 1", "Parchment, leather", "12 in x 8 in", "All three, intermingled", "1538-1769", "318"],
                ["No. 2", "Parchment, leather", "13 in x 8 in", "Baptisms, burials", "1770-1812", "196"],
                ["No. 3", "Paper, printed forms", "15 in x 9.5 in", "Marriages", "1754-1812", "144"],
                ["No. 4", "Paper, printed forms", "15 in x 10 in", "Baptisms", "1813-1837", "120"],
                ["No. 5", "Paper, printed forms", "15 in x 10 in", "Burials", "1813-1837", "100"],
                ["No. 6", "Paper, printed forms", "15 in x 10 in", "Marriages", "1813-1837", "96"],
            ],
            [1.3 * cm, 3.4 * cm, 2.5 * cm, 4.2 * cm, 2.2 * cm, 1.4 * cm],
        ),  # fmt: skip
        p("4.1 Gaps in the Record", H2),
        p(
            "No entries survive for the years 1645 to 1652. Pages 211 to 214 of book No. 1 are missing, and "
            "marriages between 1754 and 1756 appear in both book No. 1 and book No. 3."
        ),
        PageBreak(),
        # ---- page 4
        p("5 Entries Over Time", H1),
        p(
            "The number of baptisms rose steadily through the eighteenth century and peaked in the 1800s. Burials "
            "rose and fell with epidemic years, while marriages grew slowly. The chart shows totals per decade."
        ),
        image(entries_chart(), 15.5 * cm),
        p("Figure 2: Register entries per decade, 1700s-1830s", CAPTION),
        p("Table 2: Entries in the last four decades of the volume", CAPTION),
        table(decades_table, [3 * cm, 3 * cm, 3 * cm, 3 * cm]),
        PageBreak(),
        # ---- page 5
        p("6 Who Was Recorded", H1),
        p(
            "A small number of families dominate the registers. The Metcalfe family appears in every century of the "
            "volume, and lead mining families from the dale appear from the 1770s onwards."
        ),
        p("Table 3: Most frequent surnames in baptism entries, 1700-1837", CAPTION),
        table(
            [
                ["Surname", "Baptisms", "First recorded", "Last recorded"],
                ["Metcalfe", "86", "1541", "1836"],
                ["Harker", "54", "1562", "1835"],
                ["Thompson", "47", "1588", "1837"],
                ["Spence", "39", "1602", "1829"],
                ["Hutchinson", "33", "1610", "1834"],
                ["Alderson", "31", "1625", "1837"],
            ],
            [4 * cm, 3 * cm, 3.5 * cm, 3.5 * cm],
        ),  # fmt: skip
        Spacer(1, 6),
        image(occupations_chart(), 14 * cm),
        p("Figure 3: Fathers' occupations in baptism entries, 1813-1837", CAPTION),
        PageBreak(),
        # ---- page 6
        p("7 Condition and Conservation", H1),
        p(
            "A condition survey of every page was completed in 2024. Book No. 1 is the most fragile: its iron-gall "
            "ink is fading and many leaves are cockled. The paper books are mostly in good condition, although book "
            "No. 5 has water stains on 14 pages."
        ),
        p("Table 4: Condition survey by register book (2024)", CAPTION),
        table(
            [
                ["Book", "Condition", "Pages repaired in 2024", "Conservation priority"],
                ["No. 1", "Fragile: fading ink, cockled leaves", "46", "High"],
                ["No. 2", "Fair: loose binding", "12", "Medium"],
                ["No. 3", "Good", "0", "Low"],
                ["No. 4", "Good", "3", "Low"],
                ["No. 5", "Fair: water stains on 14 pages", "9", "Medium"],
                ["No. 6", "Good", "0", "Low"],
            ],
            [1.5 * cm, 6.5 * cm, 4 * cm, 3.5 * cm],
        ),  # fmt: skip
        Spacer(1, 6),
        image(condition_pie(), 12 * cm),
        p("Figure 4: Condition of surveyed pages (2024)", CAPTION),
        PageBreak(),
        # ---- page 7
        p("8 Access to the Originals", H1),
        p(
            "The original register books are kept at the North Riding Heritage Archive. The reading room is open "
            "Tuesday to Friday, 9:30 to 16:30. Books No. 1 and No. 2 are produced only by appointment, booked at "
            "least five working days in advance; the printed-form books can be seen on the day."
        ),
        p("8.1 Copies and Fees", H2),
        p("Table 5: Fees for copies of register entries", CAPTION),
        table(
            [
                ["Service", "Fee (GBP)", "Delivery"],
                ["Digital image, per page", "4.50", "Email, within 5 working days"],
                ["A4 paper print, per page", "2.00", "Post, within 10 working days"],
                ["Certified extract of one entry", "12.00", "Post, within 10 working days"],
                ["Research by archive staff, per hour", "30.00", "Report by email"],
            ],
            [6.5 * cm, 2.5 * cm, 6 * cm],
        ),  # fmt: skip
        p("8.2 Citing the Registers", H2),
        p(
            "Cite an entry by parish, register book, page and date, for example: Wensley parish register, book No. 2, "
            "page 87, baptism of 14 May 1791."
        ),
    ]


# ---------------------------------------------------------------- doc 2


def report_story(scan: Path) -> list:
    """Doc 2: Digitisation Project, Annual Report 2026. One PageBreak per page."""
    return [
        # ---- page 1
        p("Digitisation Project: Annual Report 2026", TITLE),  # one line: a 2-line title was read as a page header
        p("North Riding Heritage Archive, January 2027", CAPTION),
        p("1 Executive Summary", H1),
        p(
            "In 2026 the project digitised 2,340 of the 2,600 pages planned for the year, which is 90% of the target. "
            "OCR word accuracy averaged 91.4% across all collections. The project spent GBP 148,200 against a budget "
            "of GBP 165,000, an underspend of GBP 16,800."
        ),
        p(
            "The Wensley parish registers were completed in full. The Leyburn tithe maps fell behind schedule because "
            "the large map sheets needed a different camera setup, which arrived in September."
        ),
        p("2 Project Scope", H1),
        p(
            "The project photographs fragile local records, makes their text searchable with OCR, and publishes "
            "them online. It covers five collections:"
        ),
        bullets(
            ["Wensley and Aysgarth parish registers", "Leyburn tithe maps", "Quarter sessions papers", "Estate rentals"]
        ),  # fmt: skip
        PageBreak(),
        # ---- page 2
        p("3 A Sample Digitised Page", H1),
        p(
            "The page below, the preface of the printed Wensley register volume, is used to test OCR on typeset "
            "text. Typeset pages like this one are read almost perfectly; handwriting is much harder (section 5)."
        ),
        image(scan, 15 * cm),
        p("Figure 1: A digitised page used for OCR testing (preface, Wensley register volume)", CAPTION),
        PageBreak(),
        # ---- page 3
        p("4 Progress in 2026", H1),
        p(
            "Output was highest in early summer. It dipped in July and August, when the imaging technicians took "
            "leave and the tithe-map camera was being installed."
        ),
        p("Table 1: Pages planned and digitised per collection", CAPTION),
        table(
            [
                ["Collection", "Pages planned", "Pages digitised", "Complete", "Status"],
                ["Wensley parish registers", "1,090", "1,090", "100%", "Done"],
                ["Aysgarth parish registers", "640", "612", "96%", "On track"],
                ["Leyburn tithe maps", "180", "98", "54%", "Behind schedule"],
                ["Quarter sessions papers", "420", "360", "86%", "On track"],
                ["Estate rentals", "270", "180", "67%", "On track"],
                ["Total", "2,600", "2,340", "90%", ""],
            ],
            [5 * cm, 2.6 * cm, 2.8 * cm, 2.2 * cm, 3.4 * cm],
            total_row=True,
        ),  # fmt: skip
        Spacer(1, 6),
        image(monthly_chart(), 15.5 * cm),
        p("Figure 2: Pages digitised per month, 2026", CAPTION),
        PageBreak(),
        # ---- page 4
        p("5 OCR Quality", H1),
        p(
            "OCR accuracy depends mostly on the script. Typeset and printed pages are read almost perfectly, while "
            "older handwriting, especially secretary hand, needs manual checking. The chart shows the word accuracy "
            "measured on a sample of pages of each script type."
        ),
        image(accuracy_chart(), 15 * cm),
        p("Figure 3: OCR word accuracy by script type, 2026", CAPTION),
        p("Table 2: OCR errors in a 1,000-word sample of handwritten pages", CAPTION),
        table(
            [
                ["Error type", "Count", "Example"],
                ["Misread letters", "41", "'Metcalse' instead of 'Metcalfe'"],
                ["Merged words", "17", "'ofthe' instead of 'of the'"],
                ["Missing words", "12", "a whole line skipped"],
                ["Wrong numbers", "9", "1683 read as 1688"],
                ["Total", "79", ""],
            ],
            [4.5 * cm, 2 * cm, 8.5 * cm],
            total_row=True,
        ),  # fmt: skip
        PageBreak(),
        # ---- page 5
        p("6 Budget", H1),
        p(
            "Spending stayed within the overall budget. Conservation was the only line that went over budget, "
            "because book No. 1 of the Wensley registers needed more repairs than expected. Software costs were lower "
            "than planned after the archive moved to an open-source OCR engine."
        ),
        p("Table 3: Budget and actual spend, 2026 (GBP)", CAPTION),
        table(
            [
                ["Budget line", "Budget", "Actual", "Variance"],
                ["Staff", "92,000", "88,400", "-3,600"],
                ["Equipment", "31,000", "29,750", "-1,250"],
                ["Conservation", "18,000", "21,300", "+3,300"],
                ["Software & hosting", "14,000", "6,750", "-7,250"],
                ["Training", "10,000", "2,000", "-8,000"],
                ["Total", "165,000", "148,200", "-16,800"],
            ],
            [5 * cm, 3 * cm, 3 * cm, 3 * cm],
            total_row=True,
        ),  # fmt: skip
        Spacer(1, 6),
        image(spend_pie(), 13 * cm),
        p("Figure 4: Actual spend by category, 2026", CAPTION),
        PageBreak(),
        # ---- page 6
        p("7 Team", H1),
        p("A small core team runs the project, supported by trained volunteers who check OCR output."),
        p("Table 4: Project team, 2026", CAPTION),
        table(
            [
                ["Role", "People", "Full-time equivalent"],
                ["Project manager", "1", "1.0"],
                ["Imaging technicians", "3", "2.5"],
                ["Conservator", "1", "0.6"],
                ["OCR and metadata specialists", "2", "1.5"],
                ["Volunteers", "14", "2.1"],
            ],
            [7 * cm, 3 * cm, 4.5 * cm],
        ),  # fmt: skip
        p("8 Risks", H1),
        p("Table 5: Main project risks and how they are managed", CAPTION),
        table(
            [
                ["Risk", "Likelihood", "Impact", "Mitigation"],
                ["Bindings damaged during capture", "Medium", "High", "Conservator checks every volume first"],
                ["Scanner failure", "Low", "Medium", "Spare scanner on loan"],
                ["Low OCR accuracy (secretary hand)", "High", "Medium", "Volunteers transcribe key entries"],
                ["Conservation costs overrun again", "Medium", "Medium", "Monthly review with finance"],
            ],
            [5.6 * cm, 2.1 * cm, 1.9 * cm, 6 * cm],
        ),  # fmt: skip
        PageBreak(),
        # ---- page 7
        p("9 Plan for 2027", H1),
        p(
            "The remaining 260 pages of the 2026 plan will be finished by March 2027. The project will then start "
            "on the Swaledale parish registers, about 1,450 pages, with a budget of GBP 172,000. Every volume follows "
            "the workflow below."
        ),
        image(workflow_diagram(), 15.5 * cm),
        p("Figure 5: Digitisation workflow", CAPTION),
        p(
            "The quality check looks at a 10% sample of the images of each volume; a volume fails if more than 2% of "
            "the sampled images are blurred or cropped, and is then photographed again."
        ),
    ]


def build(path: Path, title: str, story: list) -> int:
    """Render one PDF and return its page count."""
    doc = SevenPageDocument(path, title)
    doc.build(story)
    return doc.page


def main() -> None:
    """Prepare the assets, build both PDFs and check each has exactly 7 pages."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    scan = prepare_preface_scan()
    documents = [
        (OUT_DIR / "wensley-parish-registers-guide.pdf", "The Wensley Parish Registers, 1538-1837", guide_story(scan)),
        (OUT_DIR / "register-digitisation-report-2026.pdf", "Digitisation Project: Annual Report 2026",
         report_story(scan)),
    ]  # fmt: skip
    for path, title, story in documents:
        pages = build(path, title, story)
        print(f"Wrote {path.relative_to(HERE.parent)}: {pages} pages, {path.stat().st_size // 1024} KB")
        if pages != 7:
            raise SystemExit(f"{path.name} has {pages} pages; expected 7. Adjust the content per page.")


if __name__ == "__main__":
    main()

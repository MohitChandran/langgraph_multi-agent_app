# /// script
# requires-python = ">=3.12"
# dependencies = ["reportlab==5.1.0", "matplotlib==3.11.2", "pillow==13.0.0", "numpy==2.5.4"]
# ///
"""Generate the sample document used to test and evaluate the Doc QA (RAG) agent.

Run from the repo root:
    uv run scripts/make_sample_docs.py
Output: data/samples/q3-compliance-report.pdf (data/ is git-ignored; this script is the source).

The PDF is built to exercise EVERY path of the ingestion pipeline (plan 02, section 12):
  - numbered, nested headings + PDF bookmarks   -> heading paths ("4 Cloud Vendors > 4.2 SLA Terms")
  - a small table with a caption                -> a table chunk (the plan's worked example)
  - a long table (34 rows, ~800 tokens)         -> row splitting with the header repeated in every piece
  - a bar chart WITH printed values             -> exact numbers from the vision model
  - a bar chart WITHOUT printed values          -> values marked "approximately" (estimated from the axis)
  - a flow diagram                              -> steps and arrows described in order
  - a small logo and a thin decorative banner   -> must be SKIPPED by the picture filter
  - a scanned page (text inside an image)       -> only OCR can read it
Vendor names match the business database, so later questions can combine both agents.

Deterministic: fixed data, fixed random seed, reportlab's `invariant` mode and no timestamps in
the images, so every run (on every machine) produces the same pages. The RAG eval's expected
page numbers (evals/datasets/rag_questions.json) depend on that.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageDraw, ImageFilter, ImageFont  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "samples"
ASSETS_DIR = OUT_DIR / "assets"
PDF_PATH = OUT_DIR / "q3-compliance-report.pdf"
SEED = 42

STYLES = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=STYLES["BodyText"], fontSize=10.5, leading=14, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=STYLES["Heading1"], fontSize=16, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=STYLES["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=9, textColor=colors.HexColor("#444444"), spaceAfter=10)
TITLE = ParagraphStyle("DocTitle", parent=STYLES["Title"], fontSize=22, spaceAfter=4)

# Same names as the `vendors` table in the business database.
SLA_ROWS = [
    ["Vendor", "Uptime SLA", "Credit if breached", "Critical response time"],
    ["Amazon Web Services", "99.95%", "10% of monthly fee", "15 minutes"],
    ["Microsoft Azure", "99.9%", "10% of monthly fee", "1 hour"],
    ["Google Cloud", "99.95%", "10% of monthly fee", "1 hour"],
    ["Oracle Cloud Infrastructure", "99.9%", "15% of monthly fee", "2 hours"],
    ["DigitalOcean", "99.99%", "5% of monthly fee", "4 hours"],
]
SLA_WIDTHS = [5.2 * cm, 2.6 * cm, 4.2 * cm, 4.6 * cm]
INCIDENTS = {"Q1": 9, "Q2": 11, "Q3": 14, "Q4 (forecast)": 10}
CLOUD_SPEND = {
    "Amazon Web Services": 420,
    "Microsoft Azure": 310,
    "Google Cloud": 260,
    "Oracle Cloud": 150,
    "DigitalOcean": 60,
}
REGISTER_VENDORS = [
    ("Amazon Web Services", "cloud", ["Compute hosting", "Object storage", "Managed Kubernetes"]),
    ("Microsoft Azure", "cloud", ["Data warehouse", "Identity services", "Backup vault"]),
    ("Google Cloud", "cloud", ["Analytics platform", "Machine learning APIs"]),
    ("Oracle Cloud Infrastructure", "cloud", ["ERP database hosting"]),
    ("DigitalOcean", "cloud", ["Developer sandboxes"]),
    ("Dell Technologies", "hardware", ["Laptop fleet", "Server maintenance", "Storage arrays"]),
    ("Lenovo", "hardware", ["Desktop fleet", "Docking stations"]),
    ("Cisco Systems", "hardware", ["Network equipment", "VPN appliances", "Wi-Fi access points"]),
    ("HP Inc.", "hardware", ["Printers", "Monitors"]),
    ("Deloitte", "consulting", ["ERP implementation", "Security audit"]),
    ("Accenture", "consulting", ["Data platform advisory", "Cloud migration"]),
    ("Infosys", "consulting", ["Application support", "Test automation"]),
    ("Thoughtworks", "consulting", ["Platform engineering"]),
    ("DHL Supply Chain", "logistics", ["International freight", "Warehousing"]),
    ("FedEx", "logistics", ["Express parcels"]),
    ("Blue Dart Express", "logistics", ["Domestic courier"]),
    ("Baker McKenzie", "legal", ["Contract review retainer", "Data protection advice"]),
    ("Khaitan & Co", "legal", ["Regulatory compliance advisory"]),
]
OWNERS = ["A. Rao", "M. Chen", "S. Iyer", "J. Okafor", "L. Novak"]
MONTHS = ["Jan 2027", "Feb 2027", "Mar 2027", "Apr 2027", "May 2027", "Jun 2027"]


def font_path(family: str) -> str:
    """Path of a TTF font bundled with matplotlib (exists on every machine that has matplotlib)."""
    return font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a matplotlib figure as PNG without a timestamp/software tag (keeps output deterministic)."""
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def make_logo(path: Path) -> Path:
    """A small square company logo. Tiny on the page, so the picture filter must skip it."""
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.add_patch(plt.Circle((0.5, 0.5), 0.45, color="#1a4d8f"))
    ax.text(0.5, 0.5, "AO", ha="center", va="center", fontsize=28, color="white", fontweight="bold")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    return save_figure(fig, path)


def make_banner(path: Path) -> Path:
    """A thin decorative gradient strip (aspect ratio ~28:1, no caption): must be skipped as decoration."""
    gradient = np.linspace(0, 1, 1400).reshape(1, -1)
    image = (plt.get_cmap("Blues")(np.repeat(gradient, 50, axis=0))[:, :, :3] * 255).astype(np.uint8)
    PILImage.fromarray(image).save(path)
    return path


def make_incidents_chart(path: Path) -> Path:
    """Bar chart WITH the value printed on every bar: the vision model should report exact numbers."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(list(INCIDENTS), list(INCIDENTS.values()), color=["#4c78a8", "#4c78a8", "#e45756", "#9ecae9"])
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_title("Security incidents per quarter (2026)")
    ax.set_ylabel("Incidents")
    ax.set_ylim(0, 17)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, path)


def make_spend_chart(path: Path) -> Path:
    """Bar chart WITHOUT printed values, only gridlines: values must be estimated from the axis."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [name.replace(" ", "\n", 1) for name in CLOUD_SPEND]  # "Amazon\nWeb Services": no overlapping labels
    ax.bar(labels, list(CLOUD_SPEND.values()), color="#54a24b")
    ax.set_title("Q3 2026 cloud spend by vendor")
    ax.set_ylabel("USD thousands")
    ax.set_yticks(range(0, 501, 50))
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, path)


def make_escalation_diagram(path: Path) -> Path:
    """Flow diagram of the incident escalation process: boxes, a decision and arrows."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.set_xlim(-0.2, 10.2), ax.set_ylim(0, 5), ax.axis("off")
    boxes = {
        "detect": (0.0, 3.6, "Incident\ndetected"),
        "triage": (2.5, 3.6, "Support triage\n(within 30 min)"),
        "decide": (5.0, 3.6, "Severity\ncritical?"),
        "cto": (8.0, 3.6, "Notify CTO\nwithin 1 hour"),
        "ticket": (5.0, 1.6, "Open vendor\nticket"),
        "engage": (8.0, 1.6, "Vendor engaged\n& resolves"),
        "review": (8.0, 0.1, "Post-incident review\n(within 5 days)"),
    }
    for key, (x, y, label) in boxes.items():
        style = "round,pad=0.1"
        color = "#fde2c8" if key == "decide" else "#dbe9f6"
        ax.add_patch(FancyBboxPatch((x, y), 2.0, 0.9, boxstyle=style, fc=color, ec="#333333"))
        ax.text(x + 1.0, y + 0.45, label, ha="center", va="center", fontsize=8.5)

    def arrow(start: tuple[float, float], end: tuple[float, float], label: str = "", at: tuple = (0, 0)) -> None:
        """Draw an arrow between two points, with an optional label placed at `at`."""
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#333333"))
        if label:
            ax.text(*at, label, fontsize=9, color="#b03a2e", fontweight="bold")

    arrow((2.1, 4.05), (2.5, 4.05))
    arrow((4.6, 4.05), (5.0, 4.05))
    arrow((7.1, 4.05), (8.0, 4.05), "yes", at=(7.35, 4.2))
    arrow((6.0, 3.6), (6.0, 2.5), "no", at=(6.15, 2.95))
    arrow((7.1, 2.05), (8.0, 2.05))
    arrow((9.0, 3.6), (9.0, 2.5))
    arrow((9.0, 1.6), (9.0, 1.0))
    ax.set_title("Incident escalation process", fontsize=11)
    return save_figure(fig, path)


def make_scanned_memo(path: Path) -> Path:
    """A 'scanned' audit memo: text drawn INTO an image, slightly rotated, blurred and noisy.

    There is no text layer, so only OCR can read it. Rendered at 150 dpi, A4 proportions.
    """
    width, height = 1240, 1754
    page = PILImage.new("RGB", (width, height), (246, 244, 238))
    draw = ImageDraw.Draw(page)
    title_font = ImageFont.truetype(font_path("DejaVu Sans"), 40)
    text_font = ImageFont.truetype(font_path("DejaVu Sans"), 28)
    lines = [
        ("INTERNAL AUDIT MEMO", title_font),
        ("Date: 14 October 2026", text_font),
        ("Subject: Q3 2026 vendor compliance audit", text_font),
        ("", text_font),
        ("Finding 1: Two Deloitte invoices were approved", text_font),
        ("without a purchase order.", text_font),
        ("Finding 2: The Lenovo hardware contract expired on", text_font),
        ("30 September 2026 and was not renewed.", text_font),
        ("Finding 3: DHL Supply Chain shipments were insured", text_font),
        ("for only 50% of their declared value.", text_font),
        ("", text_font),
        ("Recommendation: Procurement must review every", text_font),
        ("contract 60 days before its end date.", text_font),
        ("", text_font),
        ("Signed: Head of Internal Audit", text_font),
    ]
    y = 160
    for text, font in lines:
        draw.text((130, y), text, fill=(25, 25, 25), font=font)
        y += 62 if font is title_font else 50
    page = page.rotate(0.7, resample=PILImage.Resampling.BICUBIC, fillcolor=(246, 244, 238))
    page = page.filter(ImageFilter.GaussianBlur(0.7))
    noise = np.random.default_rng(SEED).normal(0, 9, (height, width, 1))
    noisy = np.clip(np.asarray(page, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    PILImage.fromarray(noisy).save(path)
    return path


class ReportTemplate(BaseDocTemplate):
    """A4 document with one frame per page that adds a PDF bookmark (outline entry) for every heading.

    Docling uses bookmarks first, then heading numbers, to work out heading LEVELS
    (with heading_hierarchy_options enabled), so the PDF carries both.
    """

    def __init__(self, filename: str) -> None:
        """Set page size, margins, deterministic output (invariant=1) and the page template."""
        super().__init__(filename, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                         bottomMargin=2 * cm, invariant=1, title="Q3 2026 Vendor Compliance Report",
                         author="Acme Operations")  # fmt: skip
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._page_number)])
        self._bookmarks = 0

    @staticmethod
    def _page_number(canvas, doc) -> None:  # noqa: ANN001 (reportlab callback signature)
        """Draw 'Page N' in the footer. Docling should treat it as page furniture, not body text."""
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")

    def afterFlowable(self, flowable) -> None:  # noqa: ANN001, N802 (reportlab hook name)
        """After a heading is placed on a page, add a bookmark at that position (level 0 = H1, 1 = H2)."""
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            key = f"h{self._bookmarks}"
            self._bookmarks += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0 if flowable.style.name == "H1" else 1)


def table(rows: list[list[str]], col_widths: list[float], repeat_header: bool = False) -> Table:
    """A grid table with a bold, shaded header row. repeat_header repeats it on every page it spans."""
    result = Table(rows, colWidths=col_widths, repeatRows=1 if repeat_header else 0)
    result.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return result


def register_rows() -> list[list[str]]:
    """34 rows for Appendix A (one per vendor service), built deterministically from REGISTER_VENDORS."""
    rows = [["Vendor", "Category", "Service", "Contract owner", "Next review"]]
    n = 0
    for vendor, category, services in REGISTER_VENDORS:
        for service in services:
            rows.append([vendor, category, service, OWNERS[n % len(OWNERS)], MONTHS[n % len(MONTHS)]])
            n += 1
    return rows


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    """Shorthand for a paragraph."""
    return Paragraph(text, style)


def build_story(assets: dict[str, Path]) -> list:
    """The document content, top to bottom, as reportlab flowables."""
    bullets = ListFlowable(
        [
            ListItem(p("All active vendor contracts on 30 September 2026."), leftIndent=12),
            ListItem(p("Invoices issued between 1 July and 30 September 2026."), leftIndent=12),
            ListItem(p("Security incidents reported by vendors or detected internally."), leftIndent=12),
        ],
        bulletType="bullet",
    )
    return [
        Table(
            [[Image(str(assets["logo"]), 1.5 * cm, 1.5 * cm), p("<b>Acme Operations</b><br/>Procurement Office")]],
            colWidths=[2 * cm, 15 * cm],
        ),  # fmt: skip
        Spacer(1, 4),
        Image(str(assets["banner"]), 17 * cm, 0.6 * cm),
        Spacer(1, 10),
        p("Q3 2026 Vendor Compliance Report", TITLE),
        p("Prepared by Procurement for the Finance Committee, October 2026.", CAPTION),
        p("1 Introduction", H1),
        p(
            "This report reviews how our vendors complied with contract terms, service levels and security "
            "requirements in the third quarter of 2026. It covers 18 vendors across cloud, hardware, consulting, "
            "logistics and legal services, and lists the findings of the internal audit."
        ),
        p("2 Scope", H1),
        p("The review covered:"),
        bullets,
        p("3 Vendor Policy", H1),
        p("3.1 Approval Process", H2),
        p(
            "Invoices above USD 25,000 require approval by a Finance Director before payment. Invoices without a "
            "matching purchase order must be escalated to Procurement within two business days."
        ),
        p("3.2 Payment Terms", H2),
        p(
            "Standard payment terms are Net 30. Logistics vendors (DHL Supply Chain, FedEx and Blue Dart Express) "
            "are paid on Net 15 terms, and Oracle Cloud Infrastructure and the hardware vendors Dell Technologies "
            "and Cisco Systems on Net 45. Lenovo has Net 60 terms."
        ),
        p("4 Cloud Vendors", H1),
        p("4.1 Data Residency", H2),
        p(
            "Customer data must stay in the European Union or India. Every cloud vendor must name the regions it "
            "stores data in, and must notify us 30 days before moving data to a new region."
        ),
        p("4.2 SLA Terms", H2),
        p(
            "All cloud providers must commit to a monthly uptime of at least 99.9%. Breaches are reviewed quarterly "
            "by Procurement, and service credits are claimed within 30 days of the end of the month."
        ),
        KeepTogether([p("Table 1: Cloud vendor SLA commitments", CAPTION), table(SLA_ROWS, SLA_WIDTHS)]),
        Spacer(1, 8),
        p("4.3 Exit Clauses", H2),
        p(
            "Every cloud contract must include a 90-day termination for convenience clause and a data export "
            "obligation in an open format at no extra cost."
        ),
        KeepTogether(
            [
                p("5 Security Incidents", H1),
                p(
                    "Incidents rose during the first three quarters, mostly phishing attempts against vendor "
                    "support portals. The fourth-quarter figure is a forecast."
                ),
                Image(str(assets["incidents"]), 14 * cm, 7.4 * cm),
                p("Figure 1: Security incidents per quarter", CAPTION),
            ]
        ),
        KeepTogether(
            [
                p("6 Cloud Spend", H1),
                p("Cloud spend is concentrated on the three largest providers. The chart shows the Q3 totals."),
                Image(str(assets["spend"]), 14 * cm, 8.4 * cm),
                p("Figure 2: Q3 2026 cloud spend by vendor (USD thousands)", CAPTION),
            ]
        ),
        KeepTogether(
            [
                p("7 Incident Escalation", H1),
                p("Every incident follows the escalation process below, owned by the Support team."),
                Image(str(assets["diagram"]), 15 * cm, 7.6 * cm),
                p("Figure 3: Incident escalation process", CAPTION),
            ]
        ),
        p("8 Audit Findings", H1),
        p("The signed internal audit memo is attached as a scanned page on the next page."),
        PageBreak(),
        Image(str(assets["memo"]), 16.5 * cm, 23.3 * cm),
        PageBreak(),
        p("Appendix A: Vendor Service Register", H1),
        p("Table 2: Vendor services, contract owners and next review dates", CAPTION),
        table(register_rows(), [4.6 * cm, 2.0 * cm, 5.0 * cm, 2.8 * cm, 2.6 * cm], repeat_header=True),
    ]


def main() -> None:
    """Render the images, then the PDF, and print where it went."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    assets = {
        "logo": make_logo(ASSETS_DIR / "logo.png"),
        "banner": make_banner(ASSETS_DIR / "banner.png"),
        "incidents": make_incidents_chart(ASSETS_DIR / "incidents.png"),
        "spend": make_spend_chart(ASSETS_DIR / "spend.png"),
        "diagram": make_escalation_diagram(ASSETS_DIR / "escalation.png"),
        "memo": make_scanned_memo(ASSETS_DIR / "audit-memo-scan.png"),
    }
    doc = ReportTemplate(str(PDF_PATH))
    doc.build(build_story(assets))
    print(f"Wrote {PDF_PATH.relative_to(REPO_ROOT)} ({doc.page} pages, {PDF_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

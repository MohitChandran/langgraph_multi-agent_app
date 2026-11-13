"""Unit tests for picture selection and description (app/agents/doc/vision.py) - plan 02, stage R3.

No Groq calls: describe_picture gets a fake LLM. The skip rules are checked one by one.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from PIL import Image

from app.agents.doc.models import Element
from app.agents.doc.vision import describe_picture, image_data_url, select_pictures, skip_reason


def picture(label: str | None = "bar_chart", size: tuple[int, int] = (800, 400), area: float | None = 0.15,
            caption: str = "", page: int = 1, image: str | None = "x.png") -> Element:  # fmt: skip
    """A picture element with sensible defaults (a big, labelled chart) that each test tweaks."""
    return Element("picture", caption, page, page, image_path=image, picture_label=label,
                   area_fraction=area, width_px=size[0], height_px=size[1])  # fmt: skip


@pytest.mark.parametrize(
    ("element", "reason_part"),
    [
        (picture(label="logo"), "label 'logo'"),
        (picture(label="signature"), "label 'signature'"),
        (picture(size=(120, 400)), "too small"),
        (picture(area=0.015), "covers 1.5% of the page"),
        (picture(label=None, size=(2000, 200), area=0.05), "decorative strip"),  # 10:1, both sides >= 150
        (picture(image=None), "no image"),
    ],
    ids=["logo", "signature", "narrow", "small-area", "banner", "no-image"],
)
def test_each_skip_rule(element: Element, reason_part: str) -> None:
    """Every skip rule produces a readable reason."""
    assert reason_part in skip_reason(element, min_area=0.02)


def test_kept_pictures_have_no_reason() -> None:
    """A big chart, a banner WITH a caption, and a picture from a page-less format (area unknown) are kept."""
    assert skip_reason(picture(), 0.02) is None
    assert skip_reason(picture(size=(1600, 150), caption="Timeline"), 0.02) is None
    assert skip_reason(picture(area=None), 0.02) is None


def test_cap_keeps_charts_and_diagrams_before_photos_and_counts_skips() -> None:
    """With a cap of 2: the chart and the flow chart win over earlier photos; the rest are skipped."""
    photo1, photo2 = picture(label="photograph", page=1), picture(label="photograph", page=2)
    chart, diagram = picture(label="bar_chart", page=5), picture(label="flow_chart", page=6)
    logo = picture(label="logo", page=1)
    kept, skipped = select_pictures([photo1, logo, photo2, chart, diagram], max_n=2, min_area=0.02)
    assert kept == [chart, diagram]  # returned in document order
    assert len(skipped) == 3  # logo + two photos over the cap -> pictures_skipped = 3
    assert sum("over the cap" in reason for _, reason in skipped) == 2


def test_large_images_are_shrunk_before_sending(tmp_path: Path) -> None:
    """A 4000x1000 image is sent at most 1600 px wide (Groq limits request size)."""
    path = tmp_path / "big.png"
    Image.new("RGB", (4000, 1000), "white").save(path)
    url = image_data_url(str(path))
    assert url.startswith("data:image/png;base64,")
    import base64
    import io

    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent.size == (1600, 400)


async def test_describe_picture_sends_prompt_and_image(tmp_path: Path) -> None:
    """The prompt carries the heading path and caption; the image goes along as a data URL."""
    path = tmp_path / "chart.png"
    Image.new("RGB", (300, 200), "white").save(path)
    received = []

    def fake_llm(messages):
        """Record the message and answer like the vision model."""
        received.append(messages)
        return AIMessage(content="Bar chart. Q3: 14 ")

    element = picture(caption="Figure 1: Incidents", image=str(path))
    description = await describe_picture(element, "5 Security Incidents", llm=RunnableLambda(fake_llm))

    assert description == "Bar chart. Q3: 14"
    text_part, image_part = received[0][0].content
    assert 'section "5 Security Incidents", caption "Figure 1: Incidents"' in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


async def test_a_failing_vision_call_returns_none_instead_of_raising(tmp_path: Path) -> None:
    """One broken picture must never fail the whole document."""
    path = tmp_path / "chart.png"
    Image.new("RGB", (300, 200), "white").save(path)

    def broken_llm(messages):
        """Simulate Groq failing after its retries."""
        raise RuntimeError("Groq is down")

    assert await describe_picture(picture(image=str(path)), "", llm=RunnableLambda(broken_llm)) is None

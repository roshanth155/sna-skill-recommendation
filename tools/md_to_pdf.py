"""Render the Markdown report to PDF with reportlab.

The report is authored as Markdown (docs/report.md) so it can be pasted into Word
or LaTeX unchanged; this script only exists to produce the PDF deliverable (D4).
It supports the small subset of Markdown the report uses: headings, paragraphs,
bullet and numbered lists, pipe tables, images with a caption, horizontal rules
(rendered as page breaks) and inline bold, italic and code.

Usage:  python tools/md_to_pdf.py [input.md] [output.pdf]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "docs" / "report.md"
DEFAULT_OUT = ROOT / "docs" / "report.pdf"

NAVY = colors.HexColor("#12263A")
ACCENT = colors.HexColor("#C8553D")
GREY = colors.HexColor("#5A6B7B")
RULE = colors.HexColor("#C9D1D9")
LIGHT = colors.HexColor("#F6F8FA")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.1, leading=12.4,
                      alignment=TA_JUSTIFY, spaceAfter=4.5, textColor=colors.HexColor("#1B1B1B"))
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=13, bulletIndent=3, spaceAfter=2)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, leading=19,
                    textColor=NAVY, spaceBefore=2, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=14.5,
                    textColor=NAVY, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10, leading=12,
                    textColor=ACCENT, spaceBefore=7, spaceAfter=3)
CAPTION = ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=7.6,
                         leading=9.5, textColor=GREY, spaceBefore=2, spaceAfter=7)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=6.8, leading=8.4)
CELL_H = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold",
                        textColor=colors.white)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=10)
CENTER = ParagraphStyle("center", parent=BODY, alignment=TA_CENTER)

CONTENT_W = A4[0] - 2 * 18 * mm


def inline(text: str) -> str:
    """Markdown inline markup -> reportlab mini-HTML (escaping first)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def is_number(cell: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?%?", cell.strip()))


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_table(lines: list[str], i: int):
    header = split_row(lines[i])
    i += 2                                  # skip the |---| separator line
    rows = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        rows.append(split_row(lines[i]))
        i += 1
    return header, rows, i


def auto_widths(header: list[str], rows: list[list[str]], ncol: int) -> list[float]:
    """Column widths proportional to sqrt of the average cell length.

    Equal widths make a table with a long text column (top skills, domain pairs)
    twice as tall as it needs to be, which costs pages.
    """
    weights = []
    for j in range(ncol):
        vals = [header[j]] + [r[j] for r in rows if j < len(r)]
        avg = sum(len(v) for v in vals) / max(len(vals), 1)
        weights.append(max(avg, 4.0) ** 0.5)
    total = sum(weights)
    widths = [CONTENT_W * w / total for w in weights]
    floor = 32.0
    widths = [max(w, floor) for w in widths]
    scale = CONTENT_W / sum(widths)
    return [w * scale for w in widths]


def build_table(header: list[str], rows: list[list[str]], widths=None) -> Table:
    ncol = len(header)
    data = [[Paragraph(inline(h), CELL_H) for h in header]]
    aligns = []
    for r in rows:
        row = []
        for c in r[:ncol]:
            style = CELL
            if is_number(c):
                style = ParagraphStyle("num", parent=CELL, alignment=2)
            row.append(Paragraph(inline(c), style))
        row += [Paragraph("", CELL)] * (ncol - len(row))
        data.append(row)
        aligns.append([is_number(c) for c in r[:ncol]])

    col_w = widths or auto_widths(header, rows, ncol)
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    return t


def build_image(path: Path, caption: str) -> list:
    if not path.exists():
        return [Paragraph(inline(f"[missing figure: {path.name}]"), CAPTION)]
    with PILImage.open(path) as im:
        w, h = im.size
    # wide figures (network, predicted pairs) can take more width; square or tall
    # ones are scaled down so they do not eat a page each
    ratio = w / h
    width = CONTENT_W * (0.90 if ratio > 1.3 else 0.74 if ratio > 1.05 else 0.56)
    img = Image(str(path), width=width, height=width * h / w)
    out = [Spacer(1, 4), img]
    if caption:
        out.append(Paragraph(inline(caption), CAPTION))
    return out


def render(md: str, base: Path) -> list:
    lines = md.splitlines()
    story: list = []
    para: list[str] = []
    bullets: list[str] = []
    numbered: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            story.append(Paragraph(inline(" ".join(para).strip()), BODY))
            para = []

    def flush_list():
        nonlocal bullets, numbered
        for item in bullets:
            story.append(Paragraph(inline(item), BULLET, bulletText="-"))
        for item in numbered:
            m = re.match(r"(\d+)\.\s+(.*)", item)
            if m:
                story.append(Paragraph(inline(m.group(2)), BULLET,
                                       bulletText=f"{m.group(1)}."))
        bullets, numbered = [], []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(
                r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush_para(); flush_list()
            header, rows, i = parse_table(lines, i)
            story.append(build_table(header, rows))
            story.append(Spacer(1, 6))
            continue

        if stripped.startswith("![") and "](" in stripped:
            flush_para(); flush_list()
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                cap, rel = m.group(1), m.group(2)
                story.extend(build_image((base / rel).resolve(), cap))
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            flush_para(); flush_list()
            story.append(PageSpacer())
            i += 1
            continue

        if stripped.startswith("### "):
            flush_para(); flush_list()
            story.append(Paragraph(inline(stripped[4:]), H3))
        elif stripped.startswith("## "):
            flush_para(); flush_list()
            story.append(Paragraph(inline(stripped[3:]), H2))
        elif stripped.startswith("# "):
            flush_para(); flush_list()
            story.append(Paragraph(inline(stripped[2:]), H1))
        elif not stripped:
            flush_para(); flush_list()
        elif stripped.startswith("- ") or (stripped.startswith("* ")
                                           and not stripped.startswith("**")):
            flush_para()
            if numbered:
                flush_list()
            bullets.append(stripped[2:])
        elif re.match(r"^\d+\.\s", stripped):
            flush_para()
            if bullets:
                flush_list()
            numbered.append(stripped)
        else:
            if bullets or numbered:
                flush_list()
            para.append(stripped)
        i += 1

    flush_para(); flush_list()
    return story


class PageSpacer:
    """Marks a page break without importing it everywhere."""


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    base = src.parent
    story = render(src.read_text(encoding="utf-8"), base)

    # horizontal rules in the Markdown become section spacing, not page breaks: five
    # forced breaks pushed a 15-page report to 27 pages
    story = [Spacer(1, 8) if isinstance(s, PageSpacer) else s for s in story]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(18 * mm, 9 * mm,
                          "Skill Recommendation via Link Prediction - SNA course project")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=16 * mm, bottomMargin=17 * mm,
                          title="Skill Recommendation via Link Prediction",
                          author="Social Network Analysis course project",
                          subject="Skill co-occurrence network and link prediction")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    doc.build(story)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

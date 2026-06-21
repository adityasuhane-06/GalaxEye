from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.65
RIGHT = 0.65
TOP = 0.65
BOTTOM = 0.65
BODY_SIZE = 8.5
CODE_SIZE = 7.0
LINE_STEP = 0.18


def wrap_markdown_line(line: str, width: int = 104) -> list[str]:
    if not line.strip():
        return [""]
    if line.startswith("|") or line.startswith("```"):
        return [line]
    if line.startswith("- "):
        wrapped = textwrap.wrap(line[2:], width=width - 4)
        return ["- " + wrapped[0], *["  " + part for part in wrapped[1:]]] if wrapped else [line]
    return textwrap.wrap(line, width=width) or [line]


def image_reference(line: str) -> tuple[str, str] | None:
    match = re.match(r"!\[(?P<caption>[^\]]*)\]\((?P<path>[^)]+)\)", line.strip())
    if not match:
        return None
    return match.group("caption"), match.group("path")


def flush_text_page(pdf: PdfPages, lines: list[tuple[str, str, float]]) -> None:
    if not lines:
        return
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    fig.patch.set_facecolor("white")
    y = 1.0 - TOP / PAGE_HEIGHT
    for text, family, size in lines:
        fig.text(LEFT / PAGE_WIDTH, y, text, ha="left", va="top", fontsize=size, family=family)
        y -= LINE_STEP
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    fig.patch.set_facecolor("white")
    fig.text(
        LEFT / PAGE_WIDTH,
        1.0 - TOP / PAGE_HEIGHT,
        caption or image_path.name,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
    )
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.80])
    try:
        image = plt.imread(image_path)
        ax.imshow(image)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Could not load image: {image_path}\n{exc}", ha="center", va="center")
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render(markdown_path: Path, output_path: Path) -> None:
    source = markdown_path.read_text(encoding="utf-8").splitlines()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_lines = int((PAGE_HEIGHT - TOP - BOTTOM) / LINE_STEP)
    page_lines: list[tuple[str, str, float]] = []
    in_code = False

    with PdfPages(output_path) as pdf:
        for raw_line in source:
            image = image_reference(raw_line)
            if image is not None:
                flush_text_page(pdf, page_lines)
                page_lines = []
                caption, rel_path = image
                add_image_page(pdf, markdown_path.parent / rel_path, caption)
                continue

            if raw_line.startswith("```"):
                in_code = not in_code
                continue

            for line in wrap_markdown_line(raw_line):
                family = "monospace" if in_code or line.startswith("|") else "sans-serif"
                size = CODE_SIZE if family == "monospace" else BODY_SIZE
                text = line
                if line.startswith("# "):
                    text = line[2:]
                    size = 15
                elif line.startswith("## "):
                    text = line[3:]
                    size = 12
                elif line.startswith("### "):
                    text = line[4:]
                    size = 10
                page_lines.append((text, family, size))
                if len(page_lines) >= max_lines:
                    flush_text_page(pdf, page_lines)
                    page_lines = []
        flush_text_page(pdf, page_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the markdown technical report to a simple PDF.")
    parser.add_argument("--input", default="reports/technical_report.md")
    parser.add_argument("--output", default="reports/technical_report.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(Path(args.input), Path(args.output))

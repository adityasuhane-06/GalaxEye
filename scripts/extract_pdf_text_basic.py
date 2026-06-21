from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path


def decode_pdf_literal(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == 0x5C and i + 1 < len(raw):
            i += 1
            esc = raw[i]
            mapping = {
                ord("n"): b"\n",
                ord("r"): b"\r",
                ord("t"): b"\t",
                ord("b"): b"\b",
                ord("f"): b"\f",
                ord("("): b"(",
                ord(")"): b")",
                ord("\\"): b"\\",
            }
            if esc in mapping:
                out.extend(mapping[esc])
            elif 48 <= esc <= 55:
                digits = bytes([esc])
                for _ in range(2):
                    if i + 1 < len(raw) and 48 <= raw[i + 1] <= 55:
                        i += 1
                        digits += bytes([raw[i]])
                    else:
                        break
                value = int(digits, 8)
                if 0 <= value <= 255:
                    out.append(value)
            else:
                out.append(esc)
        else:
            out.append(c)
        i += 1
    data = bytes(out)
    for encoding in ("utf-16-be", "utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("latin-1", errors="ignore")
    return text.replace("\x00", "")


def decode_pdf_hex(raw: bytes) -> str:
    cleaned = re.sub(rb"\s+", b"", raw)
    if len(cleaned) % 2:
        cleaned += b"0"
    try:
        data = bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return ""
    for encoding in ("utf-16-be", "utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("latin-1", errors="ignore")
    return text.replace("\x00", "")


def printable_score(text: str) -> float:
    if not text:
        return 0.0
    good = sum(ch.isprintable() or ch.isspace() for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    return (good / len(text)) * (0.25 + min(alpha / max(len(text), 1), 0.75))


def extract_blocks(stream: bytes) -> list[str]:
    blocks = re.findall(rb"BT(.*?)ET", stream, flags=re.S)
    if not blocks:
        blocks = [stream]
    texts: list[str] = []
    literal_re = re.compile(rb"\((?:\\.|[^\\()])*\)")
    hex_re = re.compile(rb"<([0-9A-Fa-f\s]{4,})>")
    for block in blocks:
        for match in literal_re.finditer(block):
            text = decode_pdf_literal(match.group(0)[1:-1])
            if len(text.strip()) >= 2 and printable_score(text) > 0.35:
                texts.append(text)
        for match in hex_re.finditer(block):
            text = decode_pdf_hex(match.group(1))
            if len(text.strip()) >= 2 and printable_score(text) > 0.35:
                texts.append(text)
    return texts


def extract_pdf_text(path: Path) -> str:
    data = path.read_bytes()
    chunks: list[str] = []

    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        start = max(0, match.start() - 3000)
        dictionary = data[start : match.start()]
        stream = match.group(1)
        decoded = stream
        if b"FlateDecode" in dictionary:
            try:
                decoded = zlib.decompress(stream)
            except zlib.error:
                try:
                    decoded = zlib.decompress(stream.strip())
                except zlib.error:
                    continue
        elif any(marker in dictionary for marker in (b"DCTDecode", b"JPXDecode", b"JBIG2Decode")):
            continue
        texts = extract_blocks(decoded)
        if texts:
            chunks.extend(texts)

    text = " ".join(chunks)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+")
    parser.add_argument("--out_dir", default="outputs/paper_texts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in args.pdfs:
        path = Path(pdf)
        text = extract_pdf_text(path)
        out_path = out_dir / f"{path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"{path.name}: {len(text)} chars -> {out_path}")


if __name__ == "__main__":
    main()

import re, json, hashlib, time
from pathlib import Path
from typing import Iterable

# PyMuPDF renamed its preferred import from ``fitz`` to ``pymupdf``.  Using
# the current name avoids the deprecation warning that otherwise appears
# before the pipeline has had a chance to print its status.
try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF
    import fitz
from langdetect import detect, LangDetectException

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract, Image = None, None

def infer_book_tag(path: Path) -> str:
    s = path.stem.lower()
    if any(x in s for x in ("beck", "cognitive", "cbt")):
        return "beck_cbt"
    if any(x in s for x in ("rogers", "client_centered", "client-centred", "person_centered")):
        return "rogers_pct"
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "unknown_book"

def detect_language(text: str) -> str:
    try:
        code = detect(text[:4000])
        return {"zh-cn": "zh", "zh-tw": "zh", "zh": "zh"}.get(code, code)
    except LangDetectException:
        return "und"

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def page_ocr(page, languages: str, render_scale: float, timeout: int) -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("OCR requested but pytesseract/Pillow are unavailable. Install requirements.")
    pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        return pytesseract.image_to_string(img, lang=languages, timeout=timeout)
    except Exception as e:
        raise RuntimeError(
            f"Tesseract OCR failed for language pack '{languages}'. "
            "Install the requested Tesseract language data (eng, chi_sim, chi_tra) "
            f"and check TESSERACT_CMD. Original error: {e}"
        )

def _progress(message: str) -> None:
    """Print immediately so long OCR runs remain observable in a terminal."""
    print(message, flush=True)


def read_pdf(path: Path, cfg: dict) -> list[dict]:
    if cfg["tesseract_cmd"]:
        if pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_cmd"]
    rows = []
    ocr_pages = skipped_pages = 0
    started = time.monotonic()
    with fitz.open(path) as doc:
        total_pages = len(doc)
        _progress(
            f"\n[ingest] PDF: {path.name} ({total_pages:,} pages, "
            f"{path.stat().st_size / 1024 / 1024:.1f} MiB)"
        )
        for idx, page in enumerate(doc, start=1):
            page_started = time.monotonic()
            extracted = clean_text(page.get_text("text"))
            method = "text"
            mode = cfg["ocr_mode"]
            use_ocr = mode == "always" or (
                mode == "auto" and len(extracted) < cfg["ocr_min_text_chars"]
            )
            status = f"text ({len(extracted):,} chars)"
            if use_ocr:
                status = "OCR"
                try:
                    ocr = clean_text(page_ocr(
                        page,
                        cfg["ocr_languages"],
                        cfg["ocr_render_scale"],
                        cfg["ocr_page_timeout"],
                    ))
                    ocr_pages += 1
                    if len(ocr) > len(extracted):
                        extracted, method = ocr, "ocr"
                    status += f" ({len(extracted):,} chars)"
                except Exception as exc:
                    # A single corrupt or unusually difficult page should not make a
                    # multi-hour book run lose all preceding work.
                    status += f" failed: {type(exc).__name__}: {exc}"
                    _progress(f"[ingest] WARNING page {idx}: {status}")
            if not extracted:
                skipped_pages += 1
            else:
                rows.append({
                    "book": infer_book_tag(path),
                    "book_file": path.name,
                    "page": idx,
                    "language": detect_language(extracted),
                    "extraction_method": method,
                    "text": extracted
                })

            # Print each page by default.  The moving average makes it clear that
            # OCR is advancing even when a single page takes several minutes.
            every = max(1, cfg["progress_every_pages"])
            if idx == 1 or idx == total_pages or idx % every == 0:
                elapsed = time.monotonic() - started
                average = elapsed / idx
                remaining = average * (total_pages - idx)
                _progress(
                    f"[ingest] {path.name}: page {idx:,}/{total_pages:,} | {status} | "
                    f"page {time.monotonic() - page_started:.1f}s | "
                    f"elapsed {elapsed / 60:.1f}m | ETA {remaining / 60:.1f}m"
                )
    _progress(
        f"[ingest] Finished {path.name}: kept {len(rows):,}/{total_pages:,} pages, "
        f"OCR used on {ocr_pages:,}, skipped empty {skipped_pages:,}; "
        f"{(time.monotonic() - started) / 60:.1f}m."
    )
    return rows

def read_text_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{
        "book": infer_book_tag(path), "book_file": path.name, "page": None,
        "language": detect_language(text), "extraction_method": "text_file", "text": clean_text(text)
    }]

def read_epub(path: Path) -> list[dict]:
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(str(path))
    rows, chapter_no = [], 0
    for item in book.get_items():
        if item.get_type() == 9:  # ebooklib ITEM_DOCUMENT
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = clean_text(soup.get_text(" ", strip=True))
            if text:
                chapter_no += 1
                rows.append({
                    "book": infer_book_tag(path), "book_file": path.name, "page": None,
                    "chapter": chapter_no, "language": detect_language(text),
                    "extraction_method": "epub_text", "text": text
                })
    return rows

def load_documents(input_dir: Path, cfg: dict) -> list[dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}. Create it and put your books inside.")
    files = [path for path in sorted(input_dir.rglob("*")) if path.is_file()]
    supported = [path for path in files if path.suffix.lower() in (".pdf", ".txt", ".md", ".epub")]
    _progress(
        f"[ingest] Found {len(supported):,} supported file(s) in {input_dir.resolve()}. "
        f"OCR mode={cfg['ocr_mode']}, languages={cfg['ocr_languages']}, "
        f"page timeout={cfg['ocr_page_timeout']}s."
    )
    rows = []
    for file_number, path in enumerate(supported, start=1):
        _progress(f"[ingest] Starting file {file_number}/{len(supported)}: {path.name}")
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            rows.extend(read_pdf(path, cfg))
        elif suffix in (".txt", ".md"):
            rows.extend(read_text_file(path))
        elif suffix == ".epub":
            rows.extend(read_epub(path))
        _progress(f"[ingest] File {file_number}/{len(supported)} complete; accumulated {len(rows):,} document units.")
    return rows

def make_chunks(rows: list[dict], size: int, overlap: int) -> list[dict]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require chunk size > 0 and 0 <= overlap < chunk size.")
    chunks, buffer, meta = [], "", None
    def emit(text, metadata):
        if text.strip():
            body = text.strip()
            digest = hashlib.sha1((metadata["book_file"] + body).encode("utf-8")).hexdigest()[:12]
            chunks.append({**metadata, "chunk_id": digest, "text": body})
    for row in rows:
        # Keep each page/chapter marker in the chunk text for evidence.
        marker = f'\n[FILE: {row["book_file"]}; PAGE: {row.get("page")}; LANG: {row["language"]}]\n'
        piece = marker + row["text"]
        current_meta = {k: row.get(k) for k in ("book", "book_file", "language", "extraction_method")}
        if row.get("page") is not None:
            current_meta["start_page"] = current_meta["end_page"] = row["page"]
        else:
            current_meta["start_page"] = current_meta["end_page"] = None
        if buffer and len(buffer) + len(piece) > size:
            emit(buffer, meta)
            buffer = buffer[-overlap:] if overlap else ""
            meta = current_meta
        if not buffer:
            meta = current_meta
        buffer += piece
        if meta and current_meta.get("end_page") is not None:
            meta["end_page"] = current_meta["end_page"]
        while len(buffer) >= size:
            emit(buffer[:size], meta)
            buffer = buffer[size-overlap:]
    if buffer:
        emit(buffer, meta)
    return chunks

def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

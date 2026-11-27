#!/usr/bin/env python3
"""
Convert psychotherapy conversation JSONL (and optional annotation_parts.jsonl)
into a readable DOCX document.

Install:
    pip install python-docx

Usage:
    python jsonl_to_docx.py conversations.jsonl
    python jsonl_to_docx.py conversations.jsonl --annotations annotation_parts.jsonl
    python jsonl_to_docx.py conversations.jsonl --output psychotherapy_readable.docx

Each JSONL line is treated as one conversation/annotation record.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSON objects from a JSONL file, reporting malformed line numbers."""
    records = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                print(f"Skipping {path}, line {line_number}: expected a JSON object.")
                continue
            records.append(value)
    return records


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_run_font(run, name: str = "Arial", size: Optional[float] = None,
                 bold: Optional[bool] = None, color: Optional[str] = None) -> None:
    run.font.name = name
    # Set the East Asian font too, so Chinese text displays with a suitable font.
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    r_fonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_label_value(doc: Document, label: str, value: Any) -> None:
    if value is None or value == "":
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{label}: ")
    set_run_font(r, bold=True, color="34495E")
    r = p.add_run(str(value))
    set_run_font(r)


def add_turn(doc: Document, turn: Dict[str, Any], index: int) -> None:
    speaker = str(turn.get("speaker", "unknown")).strip()
    speaker_title = speaker.replace("_", " ").title()
    turn_id = turn.get("turn_id", f"Turn {index}")

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{speaker_title}  ·  {turn_id}")
    set_run_font(r, size=11, bold=True,
                 color="1F618D" if speaker.lower() == "therapist" else "7D3C98")

    text = turn.get("text", "")
    if text:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.12
        r = p.add_run(str(text))
        set_run_font(r, size=10.5)

    for key, label in (
        ("emotional_state", "Emotional state"),
        ("belief_state", "Belief state"),
    ):
        value = turn.get(key)
        if value:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.28)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(f"{label}: ")
            set_run_font(r, size=9, bold=True, color="566573")
            r = p.add_run(str(value))
            set_run_font(r, size=9, color="566573")


def add_generic_value(doc: Document, value: Any, depth: int = 0) -> None:
    """Readable fallback renderer for annotation objects of unknown schema."""
    indent = Inches(min(0.2 * depth, 1.0))
    if isinstance(value, dict):
        for key, child in value.items():
            if child is None or child == "" or child == {} or child == []:
                continue
            if isinstance(child, (dict, list)):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = indent
                r = p.add_run(str(key).replace("_", " ").title())
                set_run_font(r, size=10, bold=True, color="34495E")
                add_generic_value(doc, child, depth + 1)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = indent
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(f"{key.replace('_', ' ').title()}: ")
                set_run_font(r, size=9, bold=True)
                r = p.add_run(str(child))
                set_run_font(r, size=9)
    elif isinstance(value, list):
        for item in value:
            if item in (None, "", {}, []):
                continue
            if isinstance(item, (dict, list)):
                add_generic_value(doc, item, depth + 1)
            else:
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.left_indent = indent
                r = p.add_run(str(item))
                set_run_font(r, size=9)
    else:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = indent
        r = p.add_run(str(value))
        set_run_font(r, size=9)


def add_conversation_record(doc: Document, record: Dict[str, Any],
                            record_index: int) -> None:
    if record_index > 1:
        doc.add_page_break()

    title = record.get("record_id", f"Conversation {record_index}")
    doc.add_heading(str(title), level=1)

    metadata = [
        ("Language", record.get("language")),
        ("Modality", record.get("modality")),
        ("Synthetic", record.get("is_synthetic")),
        ("Generation provider", record.get("generation_provider")),
        ("Generation model", record.get("generation_model")),
    ]
    for label, value in metadata:
        add_label_value(doc, label, value)

    sessions = record.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []

    for session_index, session in enumerate(sessions, start=1):
        session_number = session.get("session_number", session_index)
        doc.add_heading(f"Session {session_number}", level=2)
        add_label_value(doc, "Goal", session.get("goal"))

        turns = session.get("turns", [])
        if isinstance(turns, list):
            for turn_index, turn in enumerate(turns, start=1):
                if isinstance(turn, dict):
                    add_turn(doc, turn, turn_index)

        add_label_value(doc, "Ending state", session.get("ending_state"))

    source_refs = record.get("source_refs", [])
    if source_refs:
        doc.add_heading("Source references", level=2)
        for ref in source_refs:
            p = doc.add_paragraph(style="List Bullet")
            r = p.add_run(str(ref))
            set_run_font(r, size=9)


def add_annotations(doc: Document, annotation_records: List[Dict[str, Any]]) -> None:
    if not annotation_records:
        return

    doc.add_page_break()
    doc.add_heading("Annotation Data", level=1)
    doc.add_paragraph(
        "The annotation schema may vary. Non-empty fields are displayed below "
        "without assuming a fixed annotation structure."
    )

    for i, record in enumerate(annotation_records, start=1):
        doc.add_heading(
            f"{record.get('record_id', 'Unknown record')} — "
            f"Session {record.get('session_number', i)}",
            level=2,
        )
        for key in ("part_id", "language", "generation_provider", "generation_model"):
            if record.get(key) not in (None, ""):
                add_label_value(doc, key.replace("_", " ").title(), record[key])

        for key in ("turn_annotations", "session_summaries"):
            value = record.get(key)
            if value:
                doc.add_heading(key.replace("_", " ").title(), level=3)
                add_generic_value(doc, value)

        if record.get("source_refs"):
            doc.add_heading("Source references", level=3)
            add_generic_value(doc, record["source_refs"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert psychotherapy conversation JSONL into a readable DOCX."
    )
    parser.add_argument("input", type=Path, help="Path to conversations.jsonl")
    parser.add_argument(
        "--annotations", type=Path, default=None,
        help="Optional path to annotation_parts.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("psychotherapy_conversations.docx"),
        help="Output DOCX path (default: psychotherapy_conversations.docx)"
    )
    args = parser.parse_args()

    conversations = read_jsonl(args.input)
    if not conversations:
        raise SystemExit("No conversation records were found.")

    annotations = read_jsonl(args.annotations) if args.annotations else []

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    # Set the default Normal style.
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    doc.add_heading("Psychotherapy Conversation Dataset", level=0)
    p = doc.add_paragraph(
        f"Conversation records: {len(conversations)}"
        + (f"  |  Annotation records: {len(annotations)}" if annotations else "")
    )
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for i, record in enumerate(conversations, start=1):
        add_conversation_record(doc, record, i)

    if annotations:
        add_annotations(doc, annotations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    print(f"Created: {args.output.resolve()}")
    print(f"Conversation records processed: {len(conversations)}")
    print(f"Annotation records processed: {len(annotations)}")


if __name__ == "__main__":
    main()

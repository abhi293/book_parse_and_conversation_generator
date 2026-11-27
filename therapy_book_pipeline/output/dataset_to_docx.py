#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Psychotherapy Dataset -> DOCX Renderer
======================================

Converts the JSONL/JSON outputs of the psychotherapy generation pipeline into:
  1. One master DOCX containing all conversation records.
  2. Optional individual DOCX files, one per conversation.
  3. Dataset/methodology appendices when manifest and anxiety rubric are supplied.

Expected inputs (all optional except conversations):
  conversations.jsonl
  annotation_parts.jsonl
  conversation_annotations.jsonl
  anxiety_rubric.json
  corpus_run_manifest.json

Install:
  pip install python-docx

Example (run from your output directory):
  python dataset_to_docx.py

Explicit paths:
  python dataset_to_docx.py ^
    --conversations conversations.jsonl ^
    --annotation-parts annotation_parts.jsonl ^
    --conversation-annotations conversation_annotations.jsonl ^
    --rubric anxiety_rubric.json ^
    --manifest corpus_run_manifest.json ^
    --output-dir docx_export ^
    --individual

python dataset_to_docx.py --mode combined --conversations conversations.jsonl --annotation-parts annotation_parts.jsonl --conversation-annotations conversation_annotations.jsonl --rubric anxiety_rubric.json

--mode combined	One DOCX containing all conversations
--mode separate	One DOCX per conversation
--mode both	Combined DOCX plus individual DOCX files
If you omit --mode, it defaults to combined.

The script is schema-tolerant for annotation objects: populated fields are rendered
recursively; empty objects/lists are skipped. It does not infer or generate clinical
annotations, scores, diagnoses, or risk assessments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


# ---------------------------------------------------------------------------
# Visual theme
# ---------------------------------------------------------------------------

NAVY = "17365D"
BLUE = "2874A6"
TEAL = "117864"
PURPLE = "7D3C98"
GREY = "566573"
LIGHT_GREY = "EAF0F5"
PALE_BLUE = "EAF2F8"
PALE_TEAL = "E8F6F3"
PALE_PURPLE = "F4ECF7"
WHITE = "FFFFFF"
DARK = "222222"

# Metadata fields are shown in the front matter or provenance section rather
# than being repeated as generic annotation content.
COMMON_METADATA_KEYS = {
    "record_id", "part_id", "session_number", "language", "modality",
    "is_synthetic", "generation_provider", "generation_model",
    "annotation_granularity", "source_refs",
}


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_no}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                print(
                    f"WARNING: skipping non-object JSON at {path}:{line_no}",
                    file=sys.stderr,
                )
                continue
            records.append(obj)
    return records


def read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def nonempty(value: Any) -> bool:
    """Return False for null, blank strings, and recursively empty containers."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(nonempty(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(nonempty(v) for v in value)
    return True


def safe_filename(value: Any, fallback: str = "conversation") -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._")
    return text[:120] or fallback


def display_key(key: Any) -> str:
    return str(key).replace("_", " ").replace("-", " ").strip().title()


def display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "Not provided"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


# ---------------------------------------------------------------------------
# DOCX formatting helpers
# ---------------------------------------------------------------------------

def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start),
                          ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_run_font(run, size: Optional[float] = None, bold: Optional[bool] = None,
                 color: Optional[str] = None, italic: Optional[bool] = None) -> None:
    run.font.name = "Arial"
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    # Western and East Asian font fallbacks.
    r_fonts.set(qn("w:ascii"), "Arial")
    r_fonts.set(qn("w:hAnsi"), "Arial")
    r_fonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    set_run_font(run, size=8, color=GREY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def setup_document(doc: Document, title: str) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.12

    for style_name, size, color in (
        ("Title", 25, NAVY),
        ("Heading 1", 18, NAVY),
        ("Heading 2", 14, BLUE),
        ("Heading 3", 11, TEAL),
    ):
        style = styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    header = section.header.paragraphs[0]
    header.text = title
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        set_run_font(run, size=8, color=GREY)

    footer = section.footer.paragraphs[0]
    add_page_number(footer)


def add_rule(doc: Document, color: str = LIGHT_GREY) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(5)
    p_pr = p._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "5")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)
    p_pr.append(borders)


def add_label_value(doc: Document, label: str, value: Any,
                    indent: float = 0.0, font_size: float = 9.5) -> None:
    if not nonempty(value):
        return
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(indent)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{label}: ")
    set_run_font(r, size=font_size, bold=True, color=GREY)
    r = p.add_run(display_value(value))
    set_run_font(r, size=font_size)


def add_bullet(doc: Document, text: str, level: int = 0) -> None:
    style = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    set_run_font(r, size=9)


def add_key_value_table(doc: Document, pairs: Sequence[Tuple[str, Any]]) -> None:
    rows = [(k, v) for k, v in pairs if nonempty(v)]
    if not rows:
        return
    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, (key, value) in enumerate(rows):
        cells = table.add_row().cells
        cells[0].width = Inches(1.8)
        cells[1].width = Inches(5.3)
        cells[0].text = str(key)
        cells[1].text = display_value(value)
        set_cell_shading(cells[0], PALE_BLUE)
        for cell in cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                for run in p.runs:
                    set_run_font(run, size=8.5,
                                 bold=(cell is cells[0]),
                                 color=NAVY if cell is cells[0] else DARK)


# ---------------------------------------------------------------------------
# Generic annotation renderer
# ---------------------------------------------------------------------------

def add_generic_content(doc: Document, value: Any, depth: int = 0,
                        skip_keys: Optional[set] = None) -> int:
    """
    Render arbitrary annotation JSON recursively.
    Returns the number of non-empty leaf values rendered.
    """
    skip_keys = skip_keys or set()
    count = 0
    indent = Inches(min(depth * 0.16, 0.8))

    if isinstance(value, dict):
        for key, child in value.items():
            if key in skip_keys or not nonempty(child):
                continue
            if isinstance(child, (dict, list)):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = indent
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(display_key(key))
                set_run_font(r, size=max(8.5, 10 - depth * 0.3),
                             bold=True, color=TEAL)
                count += add_generic_content(doc, child, depth + 1, skip_keys)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = indent
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(f"{display_key(key)}: ")
                set_run_font(r, size=9, bold=True, color=GREY)
                r = p.add_run(display_value(child))
                set_run_font(r, size=9)
                count += 1

    elif isinstance(value, list):
        for item in value:
            if not nonempty(item):
                continue
            if isinstance(item, (dict, list)):
                count += add_generic_content(doc, item, depth + 1, skip_keys)
            else:
                add_bullet(doc, display_value(item), depth)
                count += 1

    elif nonempty(value):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = indent
        r = p.add_run(display_value(value))
        set_run_font(r, size=9)
        count += 1

    return count


def add_annotation_block(doc: Document, heading: str, value: Any) -> bool:
    if not nonempty(value):
        return False
    doc.add_heading(heading, level=3)
    before = len(doc.paragraphs)
    count = add_generic_content(doc, value)
    if count == 0 and len(doc.paragraphs) == before:
        p = doc.add_paragraph("No populated annotation fields.")
        for run in p.runs:
            set_run_font(run, size=9, color=GREY, italic=True)
    return True


# ---------------------------------------------------------------------------
# Conversation rendering
# ---------------------------------------------------------------------------

def add_turn(doc: Document, turn: Dict[str, Any], index: int) -> None:
    speaker = str(turn.get("speaker", "Unknown")).strip()
    speaker_lower = speaker.lower()
    speaker_label = speaker.replace("_", " ").title()
    turn_id = turn.get("turn_id", f"Turn {index}")
    accent = BLUE if speaker_lower in {"therapist", "counsellor", "counselor"} else PURPLE
    fill = PALE_BLUE if accent == BLUE else PALE_PURPLE

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_margins(cell, top=110, start=150, bottom=110, end=150)

    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(f"{speaker_label.upper()}   |   {turn_id}")
    set_run_font(r, size=9, bold=True, color=accent)

    text = turn.get("text")
    if nonempty(text):
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.13
        r = p.add_run(str(text))
        set_run_font(r, size=10.2, color=DARK)

    for key, label in (
        ("emotional_state", "Emotional state"),
        ("belief_state", "Belief state"),
    ):
        value = turn.get(key)
        if nonempty(value):
            p = cell.add_paragraph()
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(f"{label}: ")
            set_run_font(r, size=8.5, bold=True, color=GREY)
            r = p.add_run(display_value(value))
            set_run_font(r, size=8.5, color=GREY)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.space_before = Pt(0)


def index_annotation_parts(parts: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    indexed: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for part in parts:
        rid = part.get("record_id")
        session = part.get("session_number")
        if rid is None or session is None:
            continue
        try:
            key = (str(rid), int(session))
        except (TypeError, ValueError):
            continue
        indexed[key].append(part)
    return indexed


def index_conversation_annotations(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    indexed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        rid = record.get("record_id")
        if rid is not None:
            indexed[str(rid)].append(record)
    return indexed


def add_source_refs(doc: Document, refs: Any) -> None:
    if not nonempty(refs):
        return
    doc.add_heading("Source references", level=3)
    if isinstance(refs, list):
        for ref in refs:
            if nonempty(ref):
                add_bullet(doc, str(ref))
    else:
        add_generic_content(doc, refs)


def add_session(doc: Document, record_id: str, session: Dict[str, Any],
                session_index: int,
                parts_index: Dict[Tuple[str, int], List[Dict[str, Any]]]) -> None:
    number = session.get("session_number", session_index)
    try:
        number_int = int(number)
    except (TypeError, ValueError):
        number_int = session_index

    doc.add_heading(f"Session {number}", level=2)
    add_label_value(doc, "Goal", session.get("goal"))

    turns = session.get("turns", [])
    if isinstance(turns, list):
        for turn_index, turn in enumerate(turns, start=1):
            if isinstance(turn, dict):
                add_turn(doc, turn, turn_index)

    add_label_value(doc, "Ending state", session.get("ending_state"))

    # Session annotations are matched by both record_id and session_number.
    matching_parts = parts_index.get((record_id, number_int), [])
    if matching_parts:
        doc.add_heading("Session annotations", level=3)
        for part_index, part in enumerate(matching_parts, start=1):
            part_id = part.get("part_id")
            if part_id:
                add_label_value(doc, "Annotation part", part_id, indent=0.1, font_size=8.5)

            for field in ("turn_annotations", "session_summaries"):
                value = part.get(field)
                if nonempty(value):
                    add_annotation_block(doc, display_key(field), value)

            # Show any additional non-standard populated fields too.
            extra = {
                k: v for k, v in part.items()
                if k not in COMMON_METADATA_KEYS
                and k not in {"turn_annotations", "session_summaries"}
                and nonempty(v)
            }
            if extra:
                add_annotation_block(doc, "Additional session annotation fields", extra)

            if part.get("source_refs"):
                add_source_refs(doc, part["source_refs"])

        # Empty placeholders should be explicit but compact.
        if not any(
            nonempty(part.get("turn_annotations")) or
            nonempty(part.get("session_summaries")) or
            any(nonempty(v) for k, v in part.items()
                if k not in COMMON_METADATA_KEYS)
            for part in matching_parts
        ):
            p = doc.add_paragraph(
                "No populated session annotations were supplied for this session."
            )
            for run in p.runs:
                set_run_font(run, size=8.5, color=GREY, italic=True)


def add_conversation_record(
    doc: Document,
    record: Dict[str, Any],
    index: int,
    parts_index: Dict[Tuple[str, int], List[Dict[str, Any]]],
    conversation_annotation_index: Dict[str, List[Dict[str, Any]]],
    page_break_before: bool = True,
) -> None:
    if page_break_before and (doc.paragraphs or len(doc.tables)):
        doc.add_page_break()

    record_id = str(record.get("record_id", f"conversation_{index:05d}"))
    doc.add_heading(record_id, level=1)

    metadata = [
        ("Language", record.get("language")),
        ("Modality", record.get("modality")),
        ("Synthetic record", record.get("is_synthetic")),
        ("Generation provider", record.get("generation_provider")),
        ("Generation model", record.get("generation_model")),
    ]
    add_key_value_table(doc, metadata)

    sessions = record.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []

    if not sessions:
        p = doc.add_paragraph("No session data was found in this record.")
        for run in p.runs:
            set_run_font(run, color=GREY, italic=True)
    else:
        for session_index, session in enumerate(sessions, start=1):
            if isinstance(session, dict):
                add_session(doc, record_id, session, session_index, parts_index)

    if record.get("source_refs"):
        add_source_refs(doc, record["source_refs"])

    # Conversation-level annotations, matched by record_id.
    conversation_annotations = conversation_annotation_index.get(record_id, [])
    if conversation_annotations:
        doc.add_heading("Conversation-level annotations", level=2)
        for ann_index, annotation in enumerate(conversation_annotations, start=1):
            granularity = annotation.get("annotation_granularity")
            if granularity:
                add_label_value(doc, "Annotation granularity", granularity,
                                font_size=8.5)

            for field in ("turn_annotations", "session_summaries"):
                value = annotation.get(field)
                if nonempty(value):
                    add_annotation_block(doc, display_key(field), value)

            extra = {
                k: v for k, v in annotation.items()
                if k not in COMMON_METADATA_KEYS
                and k not in {"turn_annotations", "session_summaries"}
                and nonempty(v)
            }
            if extra:
                add_annotation_block(doc, "Additional conversation annotations", extra)

            if annotation.get("source_refs"):
                add_source_refs(doc, annotation["source_refs"])

        if not any(
            nonempty(a.get("turn_annotations")) or
            nonempty(a.get("session_summaries")) or
            any(nonempty(v) for k, v in a.items()
                if k not in COMMON_METADATA_KEYS)
            for a in conversation_annotations
        ):
            p = doc.add_paragraph(
                "Conversation-level annotation records exist, but their annotation "
                "payloads are empty."
            )
            for run in p.runs:
                set_run_font(run, size=8.5, color=GREY, italic=True)


# ---------------------------------------------------------------------------
# Dataset information and appendices
# ---------------------------------------------------------------------------

def add_cover_and_dataset_overview(
    doc: Document,
    conversations: List[Dict[str, Any]],
    annotation_parts: List[Dict[str, Any]],
    conversation_annotations: List[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(90)
    r = p.add_run("Psychotherapy\nConversation Dataset")
    set_run_font(r, size=27, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Readable corpus export")
    set_run_font(r, size=14, color=BLUE)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    r = p.add_run("Generated from the supplied JSONL and JSON files")
    set_run_font(r, size=9, color=GREY)

    doc.add_page_break()
    doc.add_heading("Dataset overview", level=1)

    languages = Counter(
        str(r.get("language", "Unspecified")) for r in conversations
    )
    modalities = Counter(
        str(r.get("modality", "Unspecified")) for r in conversations
    )
    providers = Counter(
        str(r.get("generation_provider", "Unspecified")) for r in conversations
    )
    models = Counter(
        str(r.get("generation_model", "Unspecified")) for r in conversations
    )
    synthetic_count = sum(1 for r in conversations if r.get("is_synthetic") is True)

    overview_pairs: List[Tuple[str, Any]] = [
        ("Conversation records", len(conversations)),
        ("Annotation-part records", len(annotation_parts)),
        ("Conversation-annotation records", len(conversation_annotations)),
        ("Synthetic records explicitly marked true", synthetic_count),
        ("Languages", ", ".join(f"{k} ({v})" for k, v in languages.most_common())),
        ("Modalities", ", ".join(f"{k} ({v})" for k, v in modalities.most_common())),
        ("Providers", ", ".join(f"{k} ({v})" for k, v in providers.most_common())),
        ("Models", ", ".join(f"{k} ({v})" for k, v in models.most_common())),
    ]

    if manifest:
        overview_pairs.extend([
            ("Manifest target count", manifest.get("target_count")),
            ("Manifest provider", manifest.get("provider")),
            ("Manifest model", manifest.get("model")),
            ("Manifest created at", manifest.get("created_at")),
            ("Manifest last updated", manifest.get("updated_at")),
        ])
        stages = manifest.get("completed_stages")
        if isinstance(stages, dict):
            overview_pairs.append((
                "Completed stages",
                "; ".join(f"{k}: {v}" for k, v in stages.items())
            ))
    add_key_value_table(doc, overview_pairs)

    if manifest:
        doc.add_heading("Run manifest details", level=2)
        excluded = {"completed_stages"}
        manifest_rest = {k: v for k, v in manifest.items()
                         if k not in excluded and nonempty(v)}
        add_generic_content(doc, manifest_rest)
        stages = manifest.get("completed_stages")
        if isinstance(stages, dict) and nonempty(stages):
            doc.add_heading("Completed pipeline stages", level=3)
            add_key_value_table(doc, [(display_key(k), v) for k, v in stages.items()])


def add_rubric_appendix(doc: Document, rubric: Optional[Dict[str, Any]]) -> None:
    if not rubric:
        return
    doc.add_page_break()
    doc.add_heading("Appendix A — Anxiety observation rubric", level=1)

    add_label_value(doc, "Rubric ID", rubric.get("rubric_id"))
    scale = rubric.get("scale")
    if isinstance(scale, dict):
        doc.add_heading("Scale definition", level=2)
        add_key_value_table(doc, [
            ("Minimum", scale.get("minimum")),
            ("Maximum", scale.get("maximum")),
            ("Interpretation", scale.get("interpretation")),
        ])

    levels = rubric.get("levels", [])
    if isinstance(levels, list) and levels:
        doc.add_heading("Interpretation levels", level=2)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = ["Range", "Label", "Description"]
        for cell, text in zip(table.rows[0].cells, headers):
            cell.text = text
            set_cell_shading(cell, NAVY)
            for run in cell.paragraphs[0].runs:
                set_run_font(run, size=8.5, bold=True, color=WHITE)
        set_repeat_table_header(table.rows[0])

        for level in levels:
            if not isinstance(level, dict):
                continue
            cells = table.add_row().cells
            range_value = level.get("range", "")
            cells[0].text = (
                f"{range_value[0]}–{range_value[1]}"
                if isinstance(range_value, list) and len(range_value) == 2
                else display_value(range_value)
            )
            cells[1].text = display_value(level.get("label", ""))
            cells[2].text = display_value(level.get("description", ""))
            for cell in cells:
                set_cell_margins(cell)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    for run in p.runs:
                        set_run_font(run, size=8.2)

    guidance = rubric.get("scoring_guidance")
    if nonempty(guidance):
        doc.add_heading("Scoring guidance", level=2)
        if isinstance(guidance, list):
            for item in guidance:
                add_bullet(doc, str(item))
        else:
            add_generic_content(doc, guidance)

    if nonempty(rubric.get("source")):
        doc.add_heading("Rubric source and qualification", level=2)
        p = doc.add_paragraph(str(rubric["source"]))
        for run in p.runs:
            set_run_font(run, size=9, color=GREY, italic=True)

    # Preserve any future/unrecognized rubric fields.
    known = {"rubric_id", "scale", "levels", "scoring_guidance", "source"}
    extras = {k: v for k, v in rubric.items()
              if k not in known and nonempty(v)}
    if extras:
        doc.add_heading("Additional rubric information", level=2)
        add_generic_content(doc, extras)


def add_provenance_appendix(doc: Document, conversations: List[Dict[str, Any]],
                            manifest: Optional[Dict[str, Any]]) -> None:
    doc.add_page_break()
    doc.add_heading("Appendix — Provenance and generation metadata", level=1)
    doc.add_paragraph(
        "The following information is reproduced from the supplied records. "
        "It describes dataset generation metadata and does not independently "
        "validate the clinical accuracy of synthetic dialogue."
    )

    if manifest:
        doc.add_heading("Corpus run manifest", level=2)
        add_generic_content(doc, manifest)

    source_counter = Counter()
    for record in conversations:
        refs = record.get("source_refs", [])
        if isinstance(refs, list):
            for ref in refs:
                if nonempty(ref):
                    source_counter[str(ref)] += 1
        elif nonempty(refs):
            source_counter[str(refs)] += 1

    if source_counter:
        doc.add_heading("Source reference inventory", level=2)
        for ref, count in source_counter.most_common():
            add_bullet(doc, f"{ref}  (referenced by {count} conversation record(s))")

    doc.add_heading("Interpretation and limitations", level=2)
    limitations = [
        "Synthetic records should be identified as synthetic wherever the source record marks them as such.",
        "Missing or empty annotation objects are displayed as unavailable; no annotation is inferred.",
        "Anxiety rubric values, where present, are reproduced as project-defined observations and are not presented as diagnoses.",
        "Source references indicate provenance metadata supplied by the pipeline; they do not by themselves establish that every generated statement is directly supported by the cited source.",
    ]
    for item in limitations:
        add_bullet(doc, item)


# ---------------------------------------------------------------------------
# Document creation
# ---------------------------------------------------------------------------

def build_master_document(
    output_path: Path,
    conversations: List[Dict[str, Any]],
    annotation_parts: List[Dict[str, Any]],
    conversation_annotations: List[Dict[str, Any]],
    rubric: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> None:
    doc = Document()
    setup_document(doc, "Psychotherapy Dataset")

    add_cover_and_dataset_overview(
        doc, conversations, annotation_parts, conversation_annotations, manifest
    )

    doc.add_page_break()
    doc.add_heading("Contents", level=1)
    p = doc.add_paragraph()
    r = p.add_run(
        "In Microsoft Word, update the table of contents field if it is not "
        "populated automatically (References → Update Table)."
    )
    set_run_font(r, size=8.5, color=GREY, italic=True)

    # Insert a real Word TOC field.
    toc_p = doc.add_paragraph()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
    toc_p._p.append(fld)

    parts_index = index_annotation_parts(annotation_parts)
    conv_ann_index = index_conversation_annotations(conversation_annotations)

    doc.add_page_break()
    doc.add_heading("Conversation records", level=1)
    for index, record in enumerate(conversations, start=1):
        add_conversation_record(
            doc, record, index, parts_index, conv_ann_index,
            page_break_before=(index > 1)
        )

    if rubric:
        add_rubric_appendix(doc, rubric)

    add_provenance_appendix(doc, conversations, manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


def build_individual_documents(
    output_dir: Path,
    conversations: List[Dict[str, Any]],
    annotation_parts: List[Dict[str, Any]],
    conversation_annotations: List[Dict[str, Any]],
    include_rubric: bool,
    rubric: Optional[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> List[Path]:
    individual_dir = output_dir / "individual_conversations"
    individual_dir.mkdir(parents=True, exist_ok=True)

    parts_index = index_annotation_parts(annotation_parts)
    conv_ann_index = index_conversation_annotations(conversation_annotations)
    created: List[Path] = []

    for index, record in enumerate(conversations, start=1):
        record_id = str(record.get("record_id", f"conversation_{index:05d}"))
        doc = Document()
        setup_document(doc, record_id)

        doc.add_heading(record_id, level=0)
        p = doc.add_paragraph("Psychotherapy conversation record")
        for run in p.runs:
            set_run_font(run, size=9, color=GREY, italic=True)

        add_conversation_record(
            doc, record, index, parts_index, conv_ann_index,
            page_break_before=False
        )

        if include_rubric and rubric:
            add_rubric_appendix(doc, rubric)

        # Compact provenance for each individual case.
        doc.add_heading("Record provenance", level=2)
        add_label_value(doc, "Record ID", record_id)
        add_label_value(doc, "Language", record.get("language"))
        add_label_value(doc, "Modality", record.get("modality"))
        add_label_value(doc, "Synthetic", record.get("is_synthetic"))
        add_label_value(doc, "Generation provider", record.get("generation_provider"))
        add_label_value(doc, "Generation model", record.get("generation_model"))
        add_source_refs(doc, record.get("source_refs"))

        filename = safe_filename(record_id) + ".docx"
        path = individual_dir / filename
        doc.save(path)
        created.append(path)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render psychotherapy JSONL/JSON dataset files into DOCX."
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "separate", "both"],
        default="combined",
        help=(
            "Export mode: combined = one DOCX containing everything; "
            "separate = one DOCX per conversation; "
            "both = combined and separate documents. "
            "Default: combined."
        )
    )
    parser.add_argument(
        "--conversations", type=Path, default=Path("conversations.jsonl"),
        help="Primary conversation JSONL file."
    )
    parser.add_argument(
        "--annotation-parts", type=Path, default=Path("annotation_parts.jsonl"),
        help="Optional session-level annotation JSONL file."
    )
    parser.add_argument(
        "--conversation-annotations", type=Path,
        default=Path("conversation_annotations.jsonl"),
        help="Optional conversation-level annotation JSONL file."
    )
    parser.add_argument(
        "--rubric", type=Path, default=Path("anxiety_rubric.json"),
        help="Optional anxiety rubric JSON file."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("corpus_run_manifest.json"),
        help="Optional corpus run manifest JSON file."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docx_export"),
        help="Directory in which DOCX files will be created."
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="Also create one DOCX per conversation."
    )
    parser.add_argument(
        "--individual-with-rubric", action="store_true",
        help="Include the rubric appendix in each individual DOCX."
    )
    parser.add_argument(
        "--master-name", default="psychotherapy_dataset_master.docx",
        help="Master DOCX filename."
    )
    return parser.parse_args()


def optional_existing(path: Path, label: str) -> Optional[Path]:
    if path.exists():
        return path
    print(f"INFO: {label} not found; skipping: {path}")
    return None



def main() -> None:
    args = parse_args()

    # Backward compatibility: old --individual flag means both.
    mode = args.mode
    if args.individual:
        if mode == "separate":
            raise SystemExit(
                "ERROR: Do not combine --individual with --mode separate. "
                "Use --mode both instead."
            )
        mode = "both"

    if not args.conversations.exists():
        raise SystemExit(
            f"ERROR: conversations file not found: {args.conversations}\n"
            "Pass the correct path using --conversations."
        )

    # Locate optional input files.
    annotation_parts_path = optional_existing(
        args.annotation_parts, "Session annotation file"
    )
    conversation_annotations_path = optional_existing(
        args.conversation_annotations, "Conversation annotation file"
    )
    rubric_path = optional_existing(args.rubric, "Anxiety rubric")
    manifest_path = optional_existing(args.manifest, "Corpus manifest")

    print("=" * 60)
    print("PSYCHOTHERAPY DATASET DOCX EXPORTER")
    print("=" * 60)
    print(f"Export mode: {mode}")

    # Load all input files.
    print("\nLoading input files...")

    conversations = read_jsonl(args.conversations)
    annotation_parts = read_jsonl(annotation_parts_path)
    conversation_annotations = read_jsonl(
        conversation_annotations_path
    )
    rubric = read_json(rubric_path)
    manifest = read_json(manifest_path)

    if not conversations:
        raise SystemExit(
            "ERROR: no conversation records were found."
        )

    # Display input summary.
    print("\nINPUT SUMMARY")
    print("-" * 40)
    print(f"Conversation records: {len(conversations)}")
    print(f"Session annotation parts: {len(annotation_parts)}")
    print(
        f"Conversation annotation records: "
        f"{len(conversation_annotations)}"
    )
    print(f"Anxiety rubric loaded: {'Yes' if rubric else 'No'}")
    print(f"Corpus manifest loaded: {'Yes' if manifest else 'No'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # COMBINED MODE
    if mode in ("combined", "both"):
        master_path = args.output_dir / args.master_name

        print("\nBuilding combined DOCX...")
        build_master_document(
            master_path,
            conversations,
            annotation_parts,
            conversation_annotations,
            rubric,
            manifest
        )

        print(f"Created: {master_path.resolve()}")

    # SEPARATE MODE
    if mode in ("separate", "both"):
        print("\nBuilding individual conversation DOCX files...")

        individual_paths = build_individual_documents(
            args.output_dir,
            conversations,
            annotation_parts,
            conversation_annotations,
            args.individual_with_rubric,
            rubric,
            manifest
        )

        print(
            f"Created {len(individual_paths)} individual DOCX file(s)."
        )
        print(
            "Folder: "
            f"{(args.output_dir / 'individual_conversations').resolve()}"
        )

    print("\n" + "=" * 60)
    print("EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    if mode in ("combined", "both"):
        print(
            "Remember to update the table of contents in Microsoft Word "
            "(References -> Update Table)."
        )



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

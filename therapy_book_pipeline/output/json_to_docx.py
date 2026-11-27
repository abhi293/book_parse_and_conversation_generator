import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def load_records(input_path: Path):
    """Load records from either a .jsonl file (one JSON per line) or a .json file."""
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try parsing as a standard JSON list/dict first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: parse line-by-line (JSONL)
    records = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON on line {line_num}: {e}")
    return records


def add_subheading(doc: Document, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11.5)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    return p


def add_label_value(doc: Document, label: str, value):
    if value is None or value == "" or value == [] or value == [{}]:
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(3)
    r_label = p.add_run(f"{label}: ")
    r_label.bold = True
    p.add_run(str(value))


def add_bullet_list(doc: Document, label: str, items: list):
    clean_items = [x for x in items if x not in (None, "", {})]
    if not clean_items:
        return
    add_subheading(doc, label)
    for item in clean_items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.add_run(str(item))


def render_blueprint_record(doc: Document, rec: dict):
    bp = rec.get("blueprint", {})
    client = bp.get("client", {})

    title = bp.get("title") or rec.get("record_id", "Untitled Record")
    heading = doc.add_heading(f"{rec.get('record_id', 'Record')} — {title}", level=1)

    meta_parts = [
        f"Language: {rec.get('language', 'N/A')}",
        f"Modality: {rec.get('modality', 'N/A')}",
        f"Synthetic: {rec.get('is_synthetic', 'N/A')}",
    ]
    add_label_value(doc, "Metadata", " | ".join(meta_parts))

    if client:
        add_subheading(doc, "Client Profile")
        profile_summary = (
            f"Pseudonym: {client.get('pseudonym', 'N/A')} | "
            f"Age: {client.get('age', 'N/A')} | "
            f"Role/Occupation: {client.get('occupation_or_role', 'N/A')}"
        )
        add_label_value(doc, "Summary", profile_summary)
        add_label_value(doc, "Cultural & Family Context", client.get("cultural_and_family_context"))
        add_bullet_list(doc, "Relevant History", client.get("relevant_history", []))
        add_bullet_list(doc, "Strengths & Supports", client.get("strengths_and_supports", []))

    add_label_value(doc, "Presenting Problem", bp.get("presenting_problem"))
    add_bullet_list(doc, "Beliefs", bp.get("beliefs", []))
    add_bullet_list(doc, "Emotions", bp.get("emotions", []))
    add_label_value(doc, "Therapy Frame", bp.get("therapy_frame"))
    add_label_value(doc, "Source Refs", ", ".join(rec.get("source_refs") or bp.get("source_refs") or []))

    doc.add_page_break()


def render_annotation_record(doc: Document, rec: dict):
    part_id = rec.get("part_id") or rec.get("record_id", "Session Record")
    session_num = rec.get("session_number", "")
    header = f"{part_id}" + (f" (Session {session_num})" if session_num != "" else "")
    doc.add_heading(header, level=1)

    add_label_value(
        doc,
        "Metadata",
        f"Record ID: {rec.get('record_id', 'N/A')} | Language: {rec.get('language', 'N/A')}",
    )

    turns = [t for t in rec.get("turn_annotations", []) if isinstance(t, dict) and t]
    for turn in turns:
        turn_id = turn.get("turn_id", "Turn")
        add_subheading(doc, f"Turn: {turn_id}")

        if turn.get("known_facts"):
            add_bullet_list(doc, "Known Facts", turn["known_facts"])
        if turn.get("client_beliefs"):
            add_bullet_list(doc, "Client Beliefs", turn["client_beliefs"])

        add_label_value(doc, "Therapist Decision", turn.get("therapist_decision"))
        add_label_value(doc, "Therapist Rationale", turn.get("therapist_rationale"))
        add_label_value(doc, "Intervention Consequence", turn.get("intervention_consequence"))
        add_label_value(doc, "Emotional Change", turn.get("emotional_change"))
        add_label_value(doc, "Confidence", turn.get("confidence"))
        add_label_value(doc, "Uncertainty", turn.get("uncertainty"))

        if turn.get("safety_flags"):
            add_bullet_list(doc, "Safety Flags", turn["safety_flags"])

    doc.add_page_break()


def convert_json_to_docx(input_file: str, output_file: str):
    input_path = Path(input_file)
    records = load_records(input_path)

    doc = Document()
    title_p = doc.add_heading(f"Exported Records: {input_path.name}", level=0)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Total records converted: {len(records)}")
    doc.add_page_break()

    for rec in records:
        if "blueprint" in rec:
            render_blueprint_record(doc, rec)
        elif "turn_annotations" in rec:
            render_annotation_record(doc, rec)
        else:
            # Generic fallback for any other JSON structure
            doc.add_heading(str(rec.get("record_id", "Record")), level=1)
            doc.add_paragraph(json.dumps(rec, indent=2, ensure_ascii=False))
            doc.add_page_break()

    # Remove trailing empty page break if present
    if len(doc.paragraphs) > 0 and doc.paragraphs[-1].text == "":
        p_element = doc.paragraphs[-1]._element
        p_element.getparent().remove(p_element)

    doc.save(output_file)
    print(f"Saved {len(records)} records to {output_file}")


if __name__ == "__main__":
    convert_json_to_docx("conversations.jsonl", "readable_records.docx")
import argparse
from pathlib import Path
from therapy_pipeline.config import get_config
from therapy_pipeline.ingest import load_documents, make_chunks, write_jsonl
from therapy_pipeline.extract import run_extraction
from therapy_pipeline.build import build_outputs
from therapy_pipeline.validate import validate

def main():
    parser = argparse.ArgumentParser(description="Bilingual CBT/PCT book parsing and dataset pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Extract PDF/EPUB/TXT text and create chunks")
    sub.add_parser("extract", help="Run LLM extraction on chunks")
    b = sub.add_parser("build", help="Deduplicate and generate final JSON datasets")
    b.add_argument("--variants-per-seed", type=int, default=3)
    sub.add_parser("validate", help="Validate generated datasets")
    sub.add_parser("all", help="Run ingest, extraction, build, validation")
    args = parser.parse_args()
    cfg = get_config()
    out = cfg["output_dir"]
    chunks_file = out / "book_chunks.jsonl"
    if args.command in ("ingest", "all"):
        docs = load_documents(cfg["input_dir"], cfg)
        chunks = make_chunks(docs, cfg["chunk_chars"], cfg["chunk_overlap"])
        write_jsonl(chunks_file, chunks)
        print(f"Read {len(docs)} document units; created {len(chunks)} chunks at {chunks_file}")
    if args.command in ("extract", "all"):
        if not chunks_file.exists():
            raise SystemExit("book_chunks.jsonl not found. Run: python main.py ingest")
        run_extraction(chunks_file, out, cfg)
    if args.command in ("build", "all"):
        variants = getattr(args, "variants_per_seed", 3)
        build_outputs(out, variants)
    if args.command in ("validate", "all"):
        ok = validate(out)
        if not ok: raise SystemExit(1)

if __name__ == "__main__":
    main()

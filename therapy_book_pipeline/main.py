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
    g = sub.add_parser("generate", help="Generate checkpointed bilingual counselling conversations")
    g.add_argument("--count", type=int, default=100, help="Total records to plan; resume skips completed records")
    g.add_argument("--batch-size", type=int, default=5)
    g.add_argument("--languages", nargs="+", default=["English", "Simplified Chinese"])
    sub.add_parser("validate-corpus", help="Validate generated conversations and annotations")
    args = parser.parse_args()
    cfg = get_config()
    out = cfg["output_dir"]
    chunks_file = out / "book_chunks.jsonl"
    if args.command in ("ingest", "all"):
        print("[pipeline] Starting ingest. Progress is printed for every page; OCR pages can take minutes.", flush=True)
        docs = load_documents(cfg["input_dir"], cfg)
        print(f"[pipeline] Chunking {len(docs):,} document units...", flush=True)
        chunks = make_chunks(docs, cfg["chunk_chars"], cfg["chunk_overlap"])
        write_jsonl(chunks_file, chunks)
        print(f"[pipeline] Read {len(docs):,} document units; created {len(chunks):,} chunks at {chunks_file}", flush=True)
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
    if args.command == "generate":
        from therapy_pipeline.corpus import generate_corpus
        if args.count < 1 or args.batch_size < 1:
            raise SystemExit("--count and --batch-size must be positive")
        generate_corpus(out, cfg, args.count, args.batch_size, args.languages)
    if args.command == "validate-corpus":
        from therapy_pipeline.corpus import validate_corpus
        if not validate_corpus(out): raise SystemExit(1)

if __name__ == "__main__":
    main()

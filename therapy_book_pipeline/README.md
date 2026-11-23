# Bilingual CBT / Person-Centered Therapy Book Pipeline

A local Python pipeline for processing your own PDF, scanned PDF, EPUB, TXT, and Markdown book files and creating:
- `strategies.json`
- `scenario_seeds.json`
- `scenarios.json`
- `anxiety_rubric.json`
- traceable intermediate extraction files

It is designed for Judith Beck's CBT material and Carl Rogers' person-centered/client-centered therapy material, but it can process other books too.

## Important limitations and research safeguards

1. OCR quality depends on scan quality and installed Tesseract language packs. Chinese OCR requires `chi_sim` and/or `chi_tra`; English requires `eng`.
2. The pipeline detects text-based versus image-based PDF pages automatically. `OCR_MODE=always` forces OCR; `OCR_MODE=auto` OCRs pages with very little embedded text.
3. LLM extraction is a candidate-generation step, not clinical validation. Review results before using them in research, training, or a counselling system.
4. Book examples are paraphrased into compact metadata. The extractor is instructed not to copy long copyrighted passages or dialogue.
5. English and Chinese editions may be translations of the same content. The pipeline retains language and source file metadata, but cross-language semantic deduplication/linking requires human review or a separate bilingual embedding stage.
6. The anxiety rubric is a project-defined observational rubric, not a validated psychometric instrument and not a rubric authored by Beck or Rogers.
7. Synthetic scenario expansions are explicitly marked `is_synthetic: true`; they must never be represented as cases from the books.
8. Do not use this pipeline to make autonomous clinical decisions. A high distress score should trigger appropriate human review and safety procedures.

## 1. Install prerequisites

### Python

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Tesseract OCR

Install Tesseract separately; `pip install pytesseract` does not install the OCR engine.

- Windows: install a Tesseract OCR Windows distribution and add its install folder to PATH.
- macOS: `brew install tesseract tesseract-lang`
- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra`

Check the installation:

```bash
tesseract --version
tesseract --list-langs
```

The language list should include the languages you intend to use: `eng`, `chi_sim`, `chi_tra`.

If Tesseract is not on PATH, set `TESSERACT_CMD` in `.env` to the full executable path, for example:

```text
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
```

## 2. Add your books

Create a `books` folder beside `main.py`. Put the books there, optionally in subfolders:

```text
books/
  beck/
    Beck_Cognitive_Behavior_Therapy_English.pdf
    Beck_Cognitive_Behavior_Therapy_Chinese.pdf
  rogers/
    Rogers_Client_Centered_Therapy_English.pdf
    Rogers_Client_Centered_Therapy_Chinese_scanned.pdf
```

Supported: `.pdf`, `.epub`, `.txt`, `.md`.

The file name is used to assign a default book tag. Include `beck`, `cognitive`, or `cbt` in Beck filenames; include `rogers`, `client_centered`, `client-centred`, or `person_centered` in Rogers filenames. Unknown files receive a tag based on their filename.

## 3. Configure the LLM

Copy `.env.example` to `.env`.

### OpenAI API

Set:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
```

API usage is billed separately from a ChatGPT subscription.

### Ollama local model

Install Ollama, pull a model capable of handling your languages and structured JSON, then set:

```bash
ollama pull qwen2.5:14b
```

In `.env`:

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
```

For Chinese and English, choose a model with good bilingual ability and enough context window for your chunk size. Local extraction quality may differ from hosted models.

## 4. Run the pipeline

Run each stage separately first:

```bash
python main.py ingest
python main.py extract
python main.py build --variants-per-seed 3
python main.py validate
```

Or run everything:

```bash
python main.py all
```

For a quick test, set `MAX_CHUNKS_PER_RUN=5` in `.env` and run `extract`. Set it back to `0` for all chunks.

## 4A. Generate a scalable bilingual conversation corpus

Install the Gemini SDK with `pip install -r requirements.txt` and set `GEMINI_API_KEY`.
After extraction/build has produced `scenario_seeds.json`, run:

```bash
python main.py generate --count 1000 --batch-size 5
python main.py validate-corpus
```

The count is a deterministic planned corpus size, not a single API batch. Each record is generated in English or Simplified Chinese, alternating by default, and contains a fictional blueprint, a multi-session conversation, and per-turn inference annotations. Use `--languages English Simplified Chinese` to set the language mix. The three JSONL files are append-only checkpoints; rerunning the same count skips records whose stages already exist and continues incomplete records. Quota failures retry with exponential backoff configured by `CORPUS_RETRIES`, `QUOTA_BACKOFF_BASE`, and `QUOTA_BACKOFF_MAX`. Start with a small count and inspect quality/cost before scaling. Each record uses three Gemini calls, so 1,000 records means roughly 3,000 calls. Source refs trace the high-level seed; generated cases and dialogue are explicitly synthetic and are not book examples.

Outputs: `scenario_blueprints.jsonl`, `conversations.jsonl`, `conversation_annotations.jsonl`. `validate-corpus` checks stage coverage, traceability, multi-session structure, and annotations. Generated text requires expert review before research or training use.

## 5. OCR settings

In `.env`:

```text
OCR_MODE=auto
OCR_MIN_TEXT_CHARS=80
PROGRESS_EVERY_PAGES=1
OCR_RENDER_SCALE=2
OCR_PAGE_TIMEOUT=300
OCR_LANGUAGES=eng+chi_sim+chi_tra
```

- `auto`: use embedded text when present; OCR pages with fewer than the threshold number of characters.
- `always`: OCR every PDF page. Useful for image PDFs with a bad hidden OCR text layer.
- `never`: never OCR; only use embedded text.

Ingestion prints a live line for each page by default, including whether it
used embedded text or OCR, elapsed time, and a moving ETA. `OCR_RENDER_SCALE=2`
is the higher-quality default; reduce it to `1.5` for a faster first pass.
`OCR_PAGE_TIMEOUT=300` ensures that an individual problematic page is reported
and skipped after five minutes while the remaining book continues. To reduce
terminal output, use `PROGRESS_EVERY_PAGES=10`.

For a book entirely in Simplified Chinese, `OCR_LANGUAGES=chi_sim`; for Traditional Chinese, `OCR_LANGUAGES=chi_tra`; for mixed Chinese and English, use `eng+chi_sim+chi_tra`. More languages can slow OCR.

## 6. Output files

| File | Purpose |
|---|---|
| `book_chunks.jsonl` | Page-aware chunks, language, file, and OCR/text extraction method |
| `strategies_raw.json` | Unmerged LLM-extracted strategies |
| `scenario_seeds.json` | Book-grounded scenario candidates |
| `extraction_failures.json` | Chunks that failed extraction |
| `strategies.json` | Deduplicated strategy catalogue |
| `scenarios.json` | Synthetic scenario variants linked to source seeds |
| `anxiety_rubric.json` | Project-defined 1–10 observational rubric |

Every extracted item should retain `source_refs` so you can trace it back to a source file, page range, and chunk.

## 7. Recommended quality-control workflow

- Inspect a sample of chunks from every book and language before running the entire collection.
- Review `extraction_failures.json`; rerun failed chunks after fixing the underlying issue.
- Check that Chinese chunks are actually Chinese and not garbled OCR.
- Compare a sample of extracted strategies against their cited source pages.
- Merge translated equivalents only after confirming that they are conceptually equivalent.
- Keep source-grounded seeds separate from synthetic expansions.
- Have a qualified mental-health professional review any clinical content before deployment.

## Troubleshooting

**`TesseractNotFoundError`**: install the Tesseract application and configure PATH or `TESSERACT_CMD`.

**Missing `chi_sim` / `chi_tra`**: install the relevant Tesseract language data. The Python package alone is insufficient.

**OCR is garbled**: use a higher-quality scan, rotate/deskew pages, use the correct Chinese language pack, or OCR one language at a time.

**OpenAI authentication error**: check `OPENAI_API_KEY`. ChatGPT Plus/Free access does not include API credits.

**Ollama connection error**: ensure the Ollama service is running and the model is pulled.

**Very large books**: process in batches using `MAX_CHUNKS_PER_RUN`; this starter implementation processes the first N chunks. For resumable large-scale processing, add a persistent per-chunk checkpoint before production use.

**Ingest appears stuck after the PyMuPDF warning**: that warning is unrelated
to the stall. Scanned PDFs can require OCR on every page, and using
`eng+chi_sim+chi_tra` at 2× rendering can take minutes per page. Update to the
latest code and rerun; the terminal will show the active page and ETA. For a
faster trial set `OCR_RENDER_SCALE=1.5`, or set `OCR_MODE=never` to verify the
embedded-text path (scanned pages will be omitted in that mode).

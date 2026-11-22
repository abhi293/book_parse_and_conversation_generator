
import json
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("book_chunks.jsonl")
BACKUP_FILE = Path("book_chunks_before_language_fix.jsonl")

# Keep English; classify every other language as Chinese.
ENGLISH_CODE = "en"
CHINESE_CODE = "zh"


def fix_languages():
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE.resolve()}")
        return

    # Create a backup before modifying anything.
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print(f"Backup created: {BACKUP_FILE.resolve()}")

    updated_lines = []
    total_chunks = 0
    english_chunks = 0
    corrected_chunks = 0
    already_chinese = 0
    other_languages = {}

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"WARNING: Invalid JSON on line {line_number}: {error}"
                )
                # Preserve malformed lines rather than silently deleting them.
                updated_lines.append(line.rstrip("\n"))
                continue

            total_chunks += 1

            old_language = chunk.get("language", "").lower().strip()

            if old_language == ENGLISH_CODE:
                english_chunks += 1

            elif old_language == CHINESE_CODE:
                already_chinese += 1

            else:
                other_languages[old_language or "(missing)"] = (
                    other_languages.get(old_language or "(missing)", 0) + 1
                )
                chunk["language"] = CHINESE_CODE
                corrected_chunks += 1

            updated_lines.append(
                json.dumps(chunk, ensure_ascii=False)
            )

    # Rewrite the JSONL file.
    with INPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("\n".join(updated_lines) + "\n")

    print("\n========== LANGUAGE FIX SUMMARY ==========")
    print(f"Total chunks:          {total_chunks}")
    print(f"English chunks kept:   {english_chunks}")
    print(f"Chinese chunks kept:   {already_chinese}")
    print(f"Chunks corrected:      {corrected_chunks}")

    if other_languages:
        print("\nOriginal non-English language values:")
        for language, count in sorted(other_languages.items()):
            print(f"  {language}: {count}")

    print("\nLanguage correction completed successfully.")


if __name__ == "__main__":
    fix_languages()

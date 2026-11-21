
import json
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import requests
from pydantic import ValidationError

from .models import ChunkExtraction


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are extracting research metadata from a user-provided book excerpt
about psychotherapy.

Use ONLY information supported by the excerpt.

Do not reproduce long passages or copyrighted dialogue.
Paraphrase concepts in your own words.

Any example_phrasing must be a short, newly written illustrative
sentence, not a quotation or close imitation of the book.

Identify:
- Explicit therapeutic strategies.
- Therapist attitudes and principles.
- Therapeutic interventions.
- Client situations, dilemmas, or case-like examples that could
  become scenario seeds.

Do not invent diagnoses, clinical facts, or claims absent from
the excerpt.

If no supported items exist, return empty arrays.

Keep source_refs exactly as the supplied source reference.

Use modality:
- cbt_beck for Judith Beck CBT material.
- person_centered_rogers for Rogers material.
- cross_modality only when the excerpt genuinely supports both.

Preserve the excerpt language in descriptions where practical.

Include language code from metadata.

Anxiety estimates are rough scenario-design estimates,
not clinical assessments.
"""


# ============================================================
# GENERAL UTILITIES
# ============================================================

def get_cfg(cfg: dict, key: str, default=None):
    """Safely retrieve configuration values."""
    value = cfg.get(key, default)

    if isinstance(value, str):
        return value.strip()

    return value


def source_ref(chunk: dict) -> str:
    """Generate a stable reference for the source chunk."""

    p = chunk.get("start_page")
    q = chunk.get("end_page")

    if p and q and p != q:
        pages = f"pp.{p}-{q}"
    elif p:
        pages = f"p.{p}"
    else:
        pages = "page_unknown"

    return (
        f'{chunk.get("book_file")}:'
        f'{pages}:'
        f'chunk={chunk["chunk_id"]}'
    )


def normalize_id(prefix: str, value: str) -> str:
    """Generate deterministic IDs."""

    value = str(value or "item")

    slug = (
        re.sub(r"[^a-z0-9]+", "_", value.lower())
        .strip("_")[:45]
        or "item"
    )

    digest = hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:6]

    return f"{prefix}_{slug}_{digest}"


def prepare_user_payload(chunk: dict) -> dict:
    """Prepare the common payload sent to every provider."""

    return {
        "source_ref": source_ref(chunk),
        "book": chunk["book"],
        "language": chunk["language"],
        "excerpt": chunk["text"],
    }


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON returned by different LLM providers.

    Handles:
    - Normal JSON.
    - Markdown JSON code fences.
    - Extra text surrounding a JSON object.
    """

    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response.")

    raw = raw.strip()

    # Remove Markdown code fences.
    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Recover a JSON object if the model added surrounding text.
    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end > start:
        candidate = raw[start:end + 1]
        return json.loads(candidate)

    raise ValueError(
        "Could not parse a valid JSON object from LLM response."
    )


# ============================================================
# OPENAI / OPENROUTER
# ============================================================

def call_openai_compatible(
    chunk: dict,
    cfg: dict,
    provider: str
) -> str:
    """
    Shared implementation for OpenAI and OpenRouter.

    Both expose OpenAI-compatible chat completion APIs.
    """

    from openai import OpenAI

    if provider == "openai":

        api_key = get_cfg(cfg, "openai_api_key")
        model = get_cfg(cfg, "openai_model")

        if not api_key:
            raise RuntimeError(
                "Missing OpenAI API key. "
                "Configure OPENAI_API_KEY."
            )

        if not model:
            raise RuntimeError(
                "Missing OPENAI_MODEL."
            )

        client = OpenAI(
            api_key=api_key,
            timeout=get_cfg(cfg, "request_timeout", 600),
            max_retries=0
        )

    elif provider == "openrouter":

        api_key = get_cfg(cfg, "openrouter_api_key")
        model = get_cfg(cfg, "openrouter_model")

        if not api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. "
                "Configure OPENROUTER_API_KEY."
            )

        if not model:
            raise RuntimeError(
                "Missing OPENROUTER_MODEL."
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=get_cfg(cfg, "request_timeout", 600),
            max_retries=0
        )

    else:
        raise ValueError(
            f"Unsupported OpenAI-compatible provider: {provider}"
        )

    user_payload = prepare_user_payload(chunk)

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False
                )
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=get_cfg(cfg, "temperature", 0.1),

        # Optional configuration.
        **(
            {"max_tokens": cfg["max_output_tokens"]}
            if cfg.get("max_output_tokens")
            else {}
        )
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            f"{provider} returned an empty completion."
        )

    return content


def call_openai(chunk: dict, cfg: dict) -> str:
    return call_openai_compatible(
        chunk,
        cfg,
        provider="openai"
    )


def call_openrouter(chunk: dict, cfg: dict) -> str:
    return call_openai_compatible(
        chunk,
        cfg,
        provider="openrouter"
    )


# ============================================================
# GOOGLE GEMINI
# ============================================================

def call_gemini(chunk: dict, cfg: dict) -> str:
    """
    Google Gemini integration using the official google-genai SDK.

    Uses structured JSON output with the Pydantic schema.
    """

    from google import genai
    from google.genai import types

    api_key = get_cfg(cfg, "gemini_api_key")
    model = get_cfg(cfg, "gemini_model")

    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. "
            "Configure GEMINI_API_KEY."
        )

    if not model:
        raise RuntimeError(
            "Missing GEMINI_MODEL."
        )

    client = genai.Client(
        api_key=api_key
    )

    user_payload = prepare_user_payload(chunk)

    response = client.models.generate_content(
        model=model,

        contents=json.dumps(
            user_payload,
            ensure_ascii=False
        ),

        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,

            response_mime_type="application/json",

            response_json_schema=ChunkExtraction.model_json_schema(),

            temperature=get_cfg(
                cfg,
                "temperature",
                0.1
            ),

            max_output_tokens=get_cfg(
                cfg,
                "max_output_tokens",
                8192
            )
        )
    )

    content = response.text

    if not content:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return content


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(chunk: dict, cfg: dict) -> str:
    """
    Local Ollama integration.

    Uses Ollama's native /api/chat endpoint and JSON schema
    constrained output.
    """

    ollama_url = get_cfg(
        cfg,
        "ollama_url",
        "http://localhost:11434"
    ).rstrip("/")

    model = get_cfg(cfg, "ollama_model")

    if not model:
        raise RuntimeError(
            "Missing OLLAMA_MODEL."
        )

    user_payload = prepare_user_payload(chunk)

    schema = ChunkExtraction.model_json_schema()

    response = requests.post(
        f"{ollama_url}/api/chat",

        json={
            "model": model,

            "stream": False,

            "format": schema,

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload,
                        ensure_ascii=False
                    )
                }
            ],

            "options": {
                "temperature": get_cfg(
                    cfg,
                    "temperature",
                    0.1
                )
            }
        },

        timeout=get_cfg(
            cfg,
            "request_timeout",
            600
        )
    )

    response.raise_for_status()

    data = response.json()

    content = data.get(
        "message",
        {}
    ).get("content")

    if not content:
        raise ValueError(
            "Ollama returned an empty response."
        )

    return content


# ============================================================
# PROVIDER REGISTRY
# ============================================================

PROVIDER_CALLERS: dict[
    str,
    Callable[[dict, dict], str]
] = {
    "openai": call_openai,
    "openrouter": call_openrouter,
    "gemini": call_gemini,
    "ollama": call_ollama,
}


def get_provider_caller(cfg: dict):
    """
    Select the appropriate provider.

    Supports:
        openai
        openrouter
        gemini
        ollama
    """

    provider = get_cfg(
        cfg,
        "provider",
        "openai"
    ).lower()

    if provider not in PROVIDER_CALLERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {', '.join(PROVIDER_CALLERS)}"
        )

    caller = PROVIDER_CALLERS[provider]

    # Validate required credentials before processing chunks.
    if provider == "openai":
        if not get_cfg(cfg, "openai_api_key"):
            raise RuntimeError(
                "Missing OpenAI API key."
            )

        if not get_cfg(cfg, "openai_model"):
            raise RuntimeError(
                "Missing OpenAI model."
            )

    elif provider == "openrouter":
        if not get_cfg(cfg, "openrouter_api_key"):
            raise RuntimeError(
                "Missing OpenRouter API key."
            )

        if not get_cfg(cfg, "openrouter_model"):
            raise RuntimeError(
                "Missing OpenRouter model."
            )

    elif provider == "gemini":
        if not get_cfg(cfg, "gemini_api_key"):
            raise RuntimeError(
                "Missing Gemini API key."
            )

        if not get_cfg(cfg, "gemini_model"):
            raise RuntimeError(
                "Missing Gemini model."
            )

    elif provider == "ollama":
        if not get_cfg(cfg, "ollama_model"):
            raise RuntimeError(
                "Missing Ollama model."
            )

    return provider, caller


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def normalize_extraction(result, chunk: dict):
    """
    Restore metadata that the LLM might have omitted.

    Also guarantees that every extracted item has a source
    reference and deterministic ID.
    """

    ref = source_ref(chunk)

    strategies = result.strategies or []
    seeds = result.scenario_seeds or []

    default_modality = {
        "beck_cbt": "cbt_beck",
        "rogers_client_centered": "person_centered_rogers",
    }.get(
        chunk.get("book"),
        "uncategorized"
    )

    for strategy in strategies:

        strategy.source_refs = list(
            dict.fromkeys(
                (strategy.source_refs or []) + [ref]
            )
        )

        if not strategy.strategy_id:
            strategy.strategy_id = normalize_id(
                "strat",
                strategy.strategy_name
            )

        if not strategy.language or strategy.language == "und":
            strategy.language = chunk["language"]

        if (
            not strategy.modality
            or strategy.modality == "uncategorized"
        ):
            strategy.modality = default_modality

    for seed in seeds:

        seed.source_refs = list(
            dict.fromkeys(
                (seed.source_refs or []) + [ref]
            )
        )

        if not seed.seed_id:
            seed.seed_id = normalize_id(
                "seed",
                seed.title
            )

        if not seed.language or seed.language == "und":
            seed.language = chunk["language"]

    return result


# ============================================================
# SINGLE CHUNK EXTRACTION
# ============================================================

def extract_chunk(
    chunk: dict,
    cfg: dict,
    retries: int = 3
):
    """
    Extract metadata from one chunk.

    Retries API failures, JSON parsing errors, and Pydantic
    validation failures.
    """

    provider, caller = get_provider_caller(cfg)

    last_error = None

    for attempt in range(retries):

        try:

            print(
                f"    Provider: {provider} | "
                f"Attempt: {attempt + 1}/{retries}"
            )

            raw = caller(chunk, cfg)

            data = parse_json_response(raw)

            result = ChunkExtraction.model_validate(data)

            result = normalize_extraction(
                result,
                chunk
            )

            return result

        except Exception as e:

            last_error = e

            if attempt < retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"    WARNING: {type(e).__name__}: {e}"
                )

                print(
                    f"    Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    raise RuntimeError(
        f"Chunk {chunk['chunk_id']} failed after "
        f"{retries} attempts using {provider}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


# ============================================================
# FULL EXTRACTION PIPELINE
# ============================================================

def run_extraction(
    chunks_path: Path,
    output_dir: Path,
    cfg: dict
):
    """
    Process all chunks from a JSONL file.

    Outputs:
        strategies_raw.json
        scenario_seeds.json
        extraction_failures.json
    """

    provider, _ = get_provider_caller(cfg)

    print("\n========================================")
    print("       PSYCHOTHERAPY EXTRACTION")
    print("========================================")

    print(f"Provider: {provider}")

    if provider == "ollama":
        print(
            f"Model: {cfg.get('ollama_model')}"
        )
    else:
        print(
            f"Model: {cfg.get(provider + '_model')}"
        )

    print("========================================\n")

    # Load chunks.
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    # Optional chunk limit.
    max_chunks = get_cfg(
        cfg,
        "max_chunks",
        0
    )

    if max_chunks and max_chunks > 0:
        chunks = chunks[:max_chunks]

    strategies = []
    seeds = []
    failures = []

    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):

        print(
            f"[{i}/{total}] Extracting "
            f"{chunk['chunk_id']} "
            f"({chunk['book']}, {chunk['language']})"
        )

        try:

            result = extract_chunk(
                chunk,
                cfg
            )

            strategies.extend([
                item.model_dump()
                for item in result.strategies
            ])

            seeds.extend([
                item.model_dump()
                for item in result.scenario_seeds
            ])

            print(
                f"    SUCCESS: "
                f"{len(result.strategies)} strategies, "
                f"{len(result.scenario_seeds)} scenario seeds"
            )

        except Exception as e:

            failure = {
                "chunk_id": chunk["chunk_id"],
                "book": chunk.get("book"),
                "language": chunk.get("language"),
                "source_ref": source_ref(chunk),
                "provider": provider,
                "error": str(e),
                "error_type": type(e).__name__,
            }

            failures.append(failure)

            print(
                f"    ERROR: {e}"
            )

    # Save results.
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {
        "strategies_raw.json": strategies,
        "scenario_seeds.json": seeds,
        "extraction_failures.json": failures,
    }

    for filename, data in outputs.items():

        output_path = output_dir / filename

        output_path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            f"Saved: {output_path}"
        )

    print("\n========================================")
    print("          EXTRACTION COMPLETE")
    print("========================================")

    print(f"Provider: {provider}")
    print(f"Processed chunks: {total}")
    print(f"Strategies extracted: {len(strategies)}")
    print(f"Scenario seeds extracted: {len(seeds)}")
    print(f"Failed chunks: {len(failures)}")

    if total:
        print(
            f"Success rate: "
            f"{((total - len(failures)) / total) * 100:.2f}%"
        )

    print("========================================\n")

    return {
        "processed_chunks": total,
        "strategies": len(strategies),
        "scenario_seeds": len(seeds),
        "failures": len(failures),
        "provider": provider,
    }

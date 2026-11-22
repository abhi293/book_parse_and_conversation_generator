
import json
import re
import hashlib
import time
import random
import threading

from pathlib import Path
from typing import Callable

import requests

from .models import ChunkExtraction


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_GEMINI_MIN_INTERVAL = 13.0
DEFAULT_GEMINI_MAX_RETRIES = 3
DEFAULT_GEMINI_MAX_SERVER_RETRY_WAIT = 300.0
DEFAULT_GENERAL_MAX_RETRIES = 3

RETRYABLE_HTTP_CODES = {
    408, 429, 500, 502, 503, 504
}

PERMANENT_HTTP_CODES = {
    400, 401, 403, 404
}


# ============================================================
# GEMINI RATE LIMITER
# ============================================================

_gemini_last_request = 0.0
_gemini_lock = threading.Lock()


def wait_for_gemini_slot(cfg: dict):
    """
    Enforce a minimum interval between Gemini requests.

    Every request, including retries, must pass through here.

    The lock also protects against simultaneous requests from
    multiple threads within this Python process.
    """

    global _gemini_last_request

    interval = float(
        get_cfg(
            cfg,
            "gemini_min_request_interval",
            DEFAULT_GEMINI_MIN_INTERVAL
        )
    )

    interval = max(0.0, interval)

    with _gemini_lock:

        now = time.monotonic()

        elapsed = now - _gemini_last_request

        remaining = interval - elapsed

        if remaining > 0:

            print(
                f"    Rate limiter: waiting "
                f"{remaining:.1f}s..."
            )

            time.sleep(remaining)

        _gemini_last_request = time.monotonic()


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


def chunk_fingerprint(chunk: dict) -> str:
    """
    Generate a fingerprint for a chunk.

    Prevents resume mode from incorrectly skipping a chunk
    whose content has changed but whose ID remains identical.
    """

    content = {
        "chunk_id": chunk.get("chunk_id"),
        "book": chunk.get("book"),
        "language": chunk.get("language"),
        "text": chunk.get("text"),
        "start_page": chunk.get("start_page"),
        "end_page": chunk.get("end_page"),
        "book_file": chunk.get("book_file"),
    }

    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()


def prepare_user_payload(chunk: dict) -> dict:
    """Prepare the common payload sent to every provider."""

    return {
        "source_ref": source_ref(chunk),
        "book": chunk["book"],
        "language": chunk["language"],
        "excerpt": chunk["text"],
    }


def atomic_write_json(path: Path, data):
    """
    Write JSON atomically.

    Prevents incomplete JSON files if the process stops
    during a write operation.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temp_path.replace(path)


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON returned by different LLM providers.

    Handles:
    - Normal JSON.
    - Markdown JSON code fences.
    - Extra text surrounding a JSON object.
    """

    if not raw or not raw.strip():
        raise ValueError(
            "LLM returned an empty response."
        )

    raw = raw.strip()

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
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, received {type(parsed).__name__}.")
        return parsed

    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end > start:

        candidate = raw[start:end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object, received {type(parsed).__name__}.")
        return parsed

    raise ValueError(
        "Could not parse a valid JSON object from LLM response."
    )


def save_raw_response(raw: str, chunk: dict, cfg: dict, attempt: int, suffix: str = "original") -> Path:
    """Save provider output for diagnosis without overwriting prior attempts."""
    output_dir = Path(get_cfg(cfg, "raw_response_dir", "output/raw_responses"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(chunk.get("chunk_id", "unknown")))
    path = output_dir / f"{safe_id}.attempt-{attempt + 1}.{suffix}.txt"
    path.write_text(raw or "", encoding="utf-8")
    return path



# ============================================================
# ERROR HANDLING
# ============================================================

def get_error_code(error: Exception):
    """
    Extract HTTP status code from provider exceptions.
    """

    code = getattr(error, "code", None)

    if code is None:
        code = getattr(error, "status_code", None)

    if code is None:

        response = getattr(
            error,
            "response",
            None
        )

        if response is not None:
            code = getattr(
                response,
                "status_code",
                None
            )

    try:
        return int(code) if code is not None else None

    except (ValueError, TypeError):
        return None


def get_error_details(error: Exception):
    """
    Retrieve structured error details when available.
    """

    details = getattr(
        error,
        "details",
        None
    )

    if callable(details):

        try:
            details = details()

        except Exception:
            details = None

    return details


def get_retry_after_header(error: Exception):
    """
    Read Retry-After from an HTTP response if available.
    """

    response = getattr(
        error,
        "response",
        None
    )

    if response is None:
        return None

    headers = getattr(
        response,
        "headers",
        {}
    )

    value = headers.get("Retry-After")

    if not value:
        return None

    try:
        return max(0.0, float(value))

    except (ValueError, TypeError):
        return None


def get_gemini_retry_delay(
    error: Exception,
    attempt: int
) -> float:
    """
    Determine how long to wait before retrying Gemini.

    Priority:
    1. Structured Google RPC RetryInfo.
    2. HTTP Retry-After header.
    3. Retry duration in the error message.
    4. Exponential backoff.

    The returned server delay is not shortened here.
    """

    fallback = min(
        60.0,
        2.0 ** attempt
    )

    details = get_error_details(error)

    if isinstance(details, dict):

        details = details.get(
            "error",
            {}
        ).get(
            "details",
            []
        )

    if isinstance(details, list):

        for item in details:

            if not isinstance(item, dict):
                continue

            type_name = item.get(
                "@type",
                ""
            )

            if type_name.endswith(
                "google.rpc.RetryInfo"
            ):

                retry_delay = item.get(
                    "retryDelay",
                    ""
                )

                match = re.search(
                    r"(\d+(?:\.\d+)?)s",
                    str(retry_delay)
                )

                if match:
                    return float(match.group(1))

    header_delay = get_retry_after_header(error)

    if header_delay is not None:
        return header_delay

    message = str(error)

    patterns = [
        r"retry in\s+(\d+(?:\.\d+)?)\s*s",
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    return fallback


def is_retryable_error(error: Exception) -> bool:
    """
    Decide whether an error should be retried.
    """

    code = get_error_code(error)

    if code in PERMANENT_HTTP_CODES:
        return False

    if code in RETRYABLE_HTTP_CODES:
        return True

    # Unknown exceptions may be transient network or parsing
    # errors, so allow the bounded retry mechanism to handle them.
    return code is None


def calculate_retry_wait(
    error: Exception,
    attempt: int,
    provider: str,
    cfg: dict
):
    """
    Calculate retry delay.

    Gemini 429 responses use Google's recommended delay.
    Other temporary errors use exponential backoff.
    """

    code = get_error_code(error)

    if provider == "gemini" and code == 429:

        wait_time = get_gemini_retry_delay(
            error,
            attempt
        )

        max_server_wait = float(
            get_cfg(
                cfg,
                "gemini_max_server_retry_wait",
                DEFAULT_GEMINI_MAX_SERVER_RETRY_WAIT
            )
        )

        # Never retry earlier than Google's recommended time.
        # If the requested delay exceeds our configured waiting
        # limit, defer the chunk to a later pipeline run.
        if wait_time > max_server_wait:

            raise RuntimeError(
                "Gemini requested a retry delay of "
                f"{wait_time:.1f}s, which exceeds the configured "
                f"maximum automatic wait of {max_server_wait:.1f}s. "
                "Chunk deferred. Run extraction again later."
            ) from error

        wait_time += random.uniform(0.5, 1.5)

        return wait_time

    wait_time = min(
        60.0,
        2.0 ** attempt
    )

    return wait_time + random.uniform(0.5, 1.5)


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
    """

    from openai import OpenAI

    if provider == "openai":

        api_key = get_cfg(
            cfg,
            "openai_api_key"
        )

        model = get_cfg(
            cfg,
            "openai_model"
        )

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
            timeout=get_cfg(
                cfg,
                "request_timeout",
                600
            ),
            max_retries=0
        )

    elif provider == "openrouter":

        api_key = get_cfg(
            cfg,
            "openrouter_api_key"
        )

        model = get_cfg(
            cfg,
            "openrouter_model"
        )

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
            timeout=get_cfg(
                cfg,
                "request_timeout",
                600
            ),
            max_retries=0
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}"
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

        temperature=get_cfg(
            cfg,
            "temperature",
            0.1
        ),

        **(
            {
                "max_tokens": cfg["max_output_tokens"]
            }
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
    Google Gemini integration using google-genai.

    Features:
    - Explicit v1beta API version.
    - SDK automatic retries disabled.
    - Centralized application-level rate limiting.
    - Structured JSON output.
    """

    from google import genai
    from google.genai import types

    api_key = get_cfg(
        cfg,
        "gemini_api_key"
    )

    model = get_cfg(
        cfg,
        "gemini_model"
    )

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
        api_key=api_key,

        http_options=types.HttpOptions(
            api_version="v1beta",

            # One SDK attempt only.
            # The application retry handler controls retries.
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )

    user_payload = prepare_user_payload(chunk)

    # Every API attempt must pass through the rate limiter.
    wait_for_gemini_slot(cfg)

    response = client.models.generate_content(
        model=model,

        contents=json.dumps(
            user_payload,
            ensure_ascii=False
        ),

        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,

            response_mime_type="application/json",

            response_json_schema=(
                ChunkExtraction.model_json_schema()
            ),

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


def repair_gemini_json(raw: str, chunk: dict, cfg: dict) -> str:
    """Request a syntax-only JSON repair from Gemini (one extra API call)."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=get_cfg(cfg, "gemini_api_key"),
        http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=1)
        )
    )
    prompt = (
        "Repair only the JSON syntax in the response below so it is valid JSON "
        "matching the supplied schema. Preserve all existing keys, values, strings, "
        "items and meanings exactly. Do not add, remove, infer, translate, summarize "
        "or rewrite content. Return only the corrected JSON object. If repair would "
        "require guessing, return an empty JSON object.\n\n"
        f"Chunk ID: {chunk.get('chunk_id', 'unknown')}\n\nRAW RESPONSE:\n{raw or ''}"
    )
    wait_for_gemini_slot(cfg)
    response = client.models.generate_content(
        model=get_cfg(cfg, "gemini_model"),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=ChunkExtraction.model_json_schema(),
            temperature=0,
            max_output_tokens=get_cfg(cfg, "max_output_tokens", 8192)
        )
    )
    if not response.text:
        raise ValueError("Gemini JSON repair returned an empty response.")
    return response.text



# ============================================================
# OLLAMA
# ============================================================

def call_ollama(chunk: dict, cfg: dict) -> str:
    """
    Local Ollama integration.
    """

    ollama_url = get_cfg(
        cfg,
        "ollama_url",
        "http://localhost:11434"
    ).rstrip("/")

    model = get_cfg(
        cfg,
        "ollama_model"
    )

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
    ).get(
        "content"
    )

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
    Select and validate the configured LLM provider.
    """

    provider = get_cfg(
        cfg,
        "provider",
        "openai"
    ).lower()

    if provider not in PROVIDER_CALLERS:

        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: "
            f"{', '.join(PROVIDER_CALLERS)}"
        )

    caller = PROVIDER_CALLERS[provider]

    required = {
        "openai": (
            "openai_api_key",
            "openai_model"
        ),
        "openrouter": (
            "openrouter_api_key",
            "openrouter_model"
        ),
        "gemini": (
            "gemini_api_key",
            "gemini_model"
        ),
        "ollama": (
            "ollama_model",
        ),
    }

    for key in required[provider]:

        if not get_cfg(cfg, key):

            raise RuntimeError(
                f"Missing configuration: {key}"
            )

    return provider, caller


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def normalize_extraction(result, chunk: dict):
    """
    Restore missing metadata and deterministic IDs.
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

        if (
            not strategy.language
            or strategy.language == "und"
        ):

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

        if (
            not seed.language
            or seed.language == "und"
        ):

            seed.language = chunk["language"]

    return result


# ============================================================
# SINGLE CHUNK EXTRACTION
# ============================================================

def extract_chunk(
    chunk: dict,
    cfg: dict,
    retries: int | None = None
):
    """
    Extract metadata from one chunk.

    Gemini retries are controlled by this function.
    SDK-level retries are disabled for Gemini.

    Permanent HTTP errors are not retried.
    Temporary API errors use exponential backoff.
    """

    provider, caller = get_provider_caller(cfg)

    if retries is None:

        if provider == "gemini":

            retries = int(
                get_cfg(
                    cfg,
                    "gemini_max_retries",
                    DEFAULT_GEMINI_MAX_RETRIES
                )
            )

        else:

            retries = int(
                get_cfg(
                    cfg,
                    "max_retries",
                    DEFAULT_GENERAL_MAX_RETRIES
                )
            )

    retries = max(1, retries)

    last_error = None

    for attempt in range(retries):

        try:

            print(
                f"    Provider: {provider} | "
                f"Attempt: {attempt + 1}/{retries}"
            )

            raw = caller(
                chunk,
                cfg
            )

            try:
                data = parse_json_response(raw)
            except (json.JSONDecodeError, ValueError) as parse_error:
                raw_path = save_raw_response(raw, chunk, cfg, attempt, "original")
                print(f"    JSON parse failed; raw response saved to: {raw_path}")

                repair_enabled = bool(get_cfg(cfg, "gemini_json_repair", True))
                if provider == "gemini" and repair_enabled:
                    try:
                        print("    Attempting one Gemini JSON syntax repair...")
                        repaired_raw = repair_gemini_json(raw, chunk, cfg)
                        repaired_path = save_raw_response(
                            repaired_raw, chunk, cfg, attempt, "repaired"
                        )
                        data = parse_json_response(repaired_raw)
                        print(f"    JSON repair succeeded; saved to: {repaired_path}")
                    except Exception as repair_error:
                        save_raw_response(
                            str(repair_error), chunk, cfg, attempt, "repair-error"
                        )
                        raise ValueError(
                            f"JSON parsing failed and repair did not recover it. "
                            f"Original response: {raw_path}. Repair error: "
                            f"{type(repair_error).__name__}: {repair_error}"
                        ) from parse_error
                else:
                    raise

            try:
                result = ChunkExtraction.model_validate(data)
            except Exception as validation_error:
                raw_path = save_raw_response(
                    raw, chunk, cfg, attempt, "schema-validation"
                )
                raise ValueError(
                    f"Response JSON failed schema validation. Raw response saved to: "
                    f"{raw_path}. Validation error: {validation_error}"
                ) from validation_error

            result = normalize_extraction(result, chunk)
            return result

        except Exception as error:

            last_error = error

            code = get_error_code(error)

            print(
                f"    WARNING: "
                f"{type(error).__name__}: {error}"
            )

            # Permanent configuration/request errors.
            if code in PERMANENT_HTTP_CODES:

                print(
                    f"    Permanent HTTP error {code}. "
                    "Not retrying."
                )

                break

            # No more attempts.
            if attempt >= retries - 1:
                break

            # Non-retryable errors.
            if not is_retryable_error(error):

                print(
                    "    Error is not retryable."
                )

                break

            try:

                wait_time = calculate_retry_wait(
                    error,
                    attempt,
                    provider,
                    cfg
                )

            except RuntimeError as deferred:

                last_error = deferred

                print(
                    f"    DEFERRED: {deferred}"
                )

                break

            print(
                f"    Retrying in "
                f"{wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Chunk {chunk['chunk_id']} failed after "
        f"{retries} attempts using {provider}. "
        f"Last error: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def empty_checkpoint():
    """
    Create an empty checkpoint structure.
    """

    return {
        "version": 1,
        "completed_chunks": {},
        "strategies": [],
        "scenario_seeds": [],
        "failures": {},
    }


def load_checkpoint(path: Path, resume: bool):
    """
    Load existing checkpoint if resume mode is enabled.
    """

    if not resume or not path.exists():

        return empty_checkpoint()

    try:

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            raise ValueError("Invalid checkpoint structure.")

        data.setdefault(
            "completed_chunks",
            {}
        )

        data.setdefault(
            "strategies",
            []
        )

        data.setdefault(
            "scenario_seeds",
            []
        )

        data.setdefault(
            "failures",
            {}
        )

        print(
            f"    Resuming checkpoint: {path}"
        )

        return data

    except Exception as error:

        raise RuntimeError(
            f"Could not load checkpoint {path}: {error}. "
            "Inspect or back up the checkpoint before continuing."
        ) from error


def save_checkpoint(
    path: Path,
    checkpoint: dict,
    output_dir: Path
):
    """
    Save checkpoint and regular output files.

    Called after every processed chunk.
    """

    atomic_write_json(
        path,
        checkpoint
    )

    write_outputs(
        output_dir,
        checkpoint
    )


def write_outputs(
    output_dir: Path,
    checkpoint: dict
):
    """
    Write the three regular pipeline output files.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {
        "strategies_raw.json": checkpoint["strategies"],

        "scenario_seeds.json": checkpoint["scenario_seeds"],

        "extraction_failures.json": list(
            checkpoint["failures"].values()
        ),
    }

    for filename, data in outputs.items():

        atomic_write_json(
            output_dir / filename,
            data
        )


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

    Additional internal file:
        .extraction_checkpoint.json

    Features:
        - Gemini v1beta.
        - Gemini rate limiting.
        - Server-directed 429 retries.
        - Exponential backoff.
        - Atomic checkpoint saving.
        - Resume successful chunks.
        - Retry previously failed chunks.
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

    if provider == "gemini":

        print(
            "API version: v1beta"
        )

        print(
            "Minimum request interval: "
            f"{get_cfg(cfg, 'gemini_min_request_interval', DEFAULT_GEMINI_MIN_INTERVAL)}s"
        )

        print(
            "Maximum retries: "
            f"{get_cfg(cfg, 'gemini_max_retries', DEFAULT_GEMINI_MAX_RETRIES)}"
        )

    print("========================================\n")

    # --------------------------------------------------------
    # LOAD CHUNKS
    # --------------------------------------------------------

    if not chunks_path.exists():

        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # OPTIONAL CHUNK LIMIT
    # --------------------------------------------------------

    max_chunks = int(
        get_cfg(
            cfg,
            "max_chunks",
            0
        ) or 0
    )

    if max_chunks > 0:

        chunks = chunks[:max_chunks]

    total = len(chunks)

    # --------------------------------------------------------
    # CHECKPOINT CONFIGURATION
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        output_dir / ".extraction_checkpoint.json"
    )

    resume = bool(
        get_cfg(
            cfg,
            "resume_extraction",
            True
        )
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        resume
    )

    completed = checkpoint["completed_chunks"]

    strategies = checkpoint["strategies"]

    seeds = checkpoint["scenario_seeds"]

    failures = checkpoint["failures"]

    print(
        f"Total chunks: {total}"
    )

    print(
        f"Previously completed: {len(completed)}"
    )

    print(
        f"Previously failed: {len(failures)}"
    )

    print(
        f"Resume mode: {'ON' if resume else 'OFF'}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS CHUNKS
    # --------------------------------------------------------

    skipped = 0
    succeeded = 0
    failed_this_run = 0

    for i, chunk in enumerate(chunks, 1):

        chunk_id = str(
            chunk["chunk_id"]
        )

        fingerprint = chunk_fingerprint(chunk)

        print(
            f"[{i}/{total}] Extracting "
            f"{chunk_id} "
            f"({chunk['book']}, {chunk['language']})"
        )

        # Skip only when the fingerprint matches.
        previous_fingerprint = completed.get(
            chunk_id
        )

        if (
            resume
            and previous_fingerprint == fingerprint
        ):

            skipped += 1

            print(
                "    SKIPPED: Already successfully extracted."
            )

            continue

        # If content changed, old completion is invalid.
        if previous_fingerprint is not None:

            completed.pop(
                chunk_id,
                None
            )

            print(
                "    Chunk content changed. Re-extracting."
            )

        try:

            result = extract_chunk(
                chunk,
                cfg
            )

            new_strategies = [
                item.model_dump()
                for item in result.strategies
            ]

            new_seeds = [
                item.model_dump()
                for item in result.scenario_seeds
            ]

            # Add successful results.
            strategies.extend(
                new_strategies
            )

            seeds.extend(
                new_seeds
            )

            # Mark successful even if arrays are empty.
            completed[chunk_id] = fingerprint

            # Remove an old failure if this retry succeeded.
            failures.pop(
                chunk_id,
                None
            )

            succeeded += 1

            print(
                f"    SUCCESS: "
                f"{len(new_strategies)} strategies, "
                f"{len(new_seeds)} scenario seeds"
            )

        except Exception as error:

            failed_this_run += 1

            failures[chunk_id] = {
                "chunk_id": chunk_id,
                "book": chunk.get("book"),
                "language": chunk.get("language"),
                "source_ref": source_ref(chunk),
                "provider": provider,
                "error": str(error),
                "error_type": type(error).__name__,
                "fingerprint": fingerprint,
            }

            print(
                f"    ERROR: {error}"
            )

        # ----------------------------------------------------
        # SAVE AFTER EVERY CHUNK
        # ----------------------------------------------------

        try:

            save_checkpoint(
                checkpoint_path,
                checkpoint,
                output_dir
            )

        except Exception as save_error:

            raise RuntimeError(
                f"Could not save extraction progress: {save_error}"
            ) from save_error

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_completed = len(completed)

    total_failures = len(failures)

    success_rate = (
        (total_completed / total) * 100
        if total
        else 0.0
    )

    print("\n========================================")
    print("          EXTRACTION COMPLETE")
    print("========================================")

    print(
        f"Provider: {provider}"
    )

    print(
        f"Chunks in this run: {total}"
    )

    print(
        f"Newly successful: {succeeded}"
    )

    print(
        f"Skipped (already completed): {skipped}"
    )

    print(
        f"Failed in this run: {failed_this_run}"
    )

    print(
        f"Total completed in checkpoint: {total_completed}"
    )

    print(
        f"Remaining failed chunks: {total_failures}"
    )

    print(
        f"Strategies extracted: {len(strategies)}"
    )

    print(
        f"Scenario seeds extracted: {len(seeds)}"
    )

    print(
        f"Success rate: {success_rate:.2f}%"
    )

    print("========================================")

    print(
        f"\nCheckpoint: {checkpoint_path}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print("========================================\n")

    return {
        "processed_chunks": total,
        "newly_successful": succeeded,
        "skipped": skipped,
        "strategies": len(strategies),
        "scenario_seeds": len(seeds),
        "failures": total_failures,
        "provider": provider,
    }
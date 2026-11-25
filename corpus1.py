# """Checkpointed bilingual synthetic counselling corpus generation."""
# import json
# import random
# import time
# from pathlib import Path


# def _read_json(path):
#     return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# def _read_jsonl(path):
#     if not path.exists():
#         return []
#     return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# def _append(path, row):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("a", encoding="utf-8") as stream:
#         stream.write(json.dumps(row, ensure_ascii=False) + "\n")
#         stream.flush()


# def _call_model(prompt, cfg, schema):
#     if cfg.get("provider", "gemini").lower() == "ollama":
#         import requests
#         url = cfg.get("ollama_url", "http://localhost:11434").rstrip("/")
#         model = cfg.get("ollama_model", "")
#         if not model:
#             raise RuntimeError("OLLAMA_MODEL is empty")
#         response = requests.post(
#             f"{url}/api/generate",
#             json={"model": model, "prompt": prompt, "stream": False,
#                   "format": "json", "options": {"temperature": cfg["temperature"]}},
#             timeout=cfg["request_timeout"],
#         )
#         response.raise_for_status()
#         raw = response.json().get("response", "")
#         if not raw:
#             raise RuntimeError("Ollama returned an empty response")
#         return json.loads(raw)
#     if cfg.get("provider", "gemini").lower() != "gemini":
#         raise RuntimeError(f"Corpus generation does not support provider '{cfg['provider']}'. Use gemini or ollama.")
#     try:
#         from google import genai
#         from google.genai import types
#     except ImportError as exc:
#         raise RuntimeError("Gemini SDK missing. Install with: pip install google-genai") from exc
#     client = genai.Client(api_key=cfg["gemini_api_key"])
#     response = client.models.generate_content(
#         model=cfg["gemini_model"], contents=prompt,
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json", response_json_schema=schema,
#             temperature=cfg["temperature"], max_output_tokens=cfg["max_output_tokens"],
#         ),
#     )
#     if not response.text:
#         raise RuntimeError("Gemini returned an empty response")
#     return json.loads(response.text)


# BLUEPRINT_SCHEMA = {"type": "OBJECT", "properties": {
#     "title": {"type": "STRING"}, "client": {"type": "OBJECT"},
#     "presenting_problem": {"type": "STRING"}, "beliefs": {"type": "ARRAY", "items": {"type": "STRING"}},
#     "emotions": {"type": "ARRAY", "items": {"type": "STRING"}},
#     "therapy_frame": {"type": "STRING"}, "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
# }}
# CONVERSATION_SCHEMA = {"type": "OBJECT", "properties": {
#     "sessions": {"type": "ARRAY", "items": {"type": "OBJECT"}},
#     "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
# }}
# ANNOTATION_SCHEMA = {"type": "OBJECT", "properties": {
#     "turn_annotations": {"type": "ARRAY", "items": {"type": "OBJECT"}},
#     "session_summaries": {"type": "ARRAY", "items": {"type": "OBJECT"}},
# }}


# def _request(prompt, cfg, schema, label):
#     for attempt in range(1, cfg["corpus_retries"] + 1):
#         try:
#             print(f"  {label}: Gemini request {attempt}/{cfg['corpus_retries']}", flush=True)
#             return _call_model(prompt, cfg, schema)
#         except Exception as exc:
#             if attempt == cfg["corpus_retries"]:
#                 raise
#             delay = min(cfg["quota_backoff_max"], cfg["quota_backoff_base"] * 2 ** (attempt - 1))
#             print(f"  request failed ({type(exc).__name__}: {exc}); wait {delay}s", flush=True)
#             time.sleep(delay)


# def _blueprint_prompt(seed, language):
#     return ("Create one original fictional counselling case blueprint in " + language +
#         ". Return JSON matching schema. Client profile must be internally consistent, include nuanced "
#         "history, strengths, context, multiple competing beliefs, emotions and interpersonal dynamics. "
#         "Use source only as a high-level therapy concept anchor; do not reproduce book text or imply the "
#         "fictional client appears in a book. Mark uncertainty. Maintain safe, non-diagnostic framing.\n" +
#         json.dumps(seed, ensure_ascii=False))


# def _conversation_prompt(blueprint, language, length, modality):
#     return (f"Write a realistic fictional multi-session counselling conversation in {language}. "
#         f"Modality: {modality}; length profile: {length}. Return JSON with sessions, each having session_number, "
#         "goal, turns array of {turn_id, speaker(client|therapist), text, emotional_state, belief_state}, "
#         "and ending_state. Include 3-6 sessions, varied turn counts, client agency, ruptures/misunderstandings, "
#         "unresolved material, and intervention consequences. CBT uses collaborative formulation and careful "
#         "experiments; person-centered uses empathic reflection and non-directivity. Do not claim clinical efficacy, "
#         "avoid diagnosing, handle risk disclosures by prioritizing immediate human safety assessment. "
#         "Keep dialogue original, not copied from source.\n" + json.dumps(blueprint, ensure_ascii=False))


# def _annotation_prompt(conversation, blueprint, language):
#     return (f"Annotate this fictional counselling corpus in {language}. Return full inference annotations: "
#         "for every turn record known facts, hypotheses (at least two competing interpretations where plausible), "
#         "supporting evidence turn IDs, contradictions, confidence 0..1, uncertainty, client beliefs, therapist "
#         "decision and rationale, intervention consequence, emotional change, safety flags. Distinguish stated, "
#         "inferred, uncertain, contradicted. Never present hypotheses as facts. Add session summaries.\n" +
#         json.dumps({"blueprint": blueprint, "conversation": conversation}, ensure_ascii=False))


# def generate_corpus(output_dir: Path, cfg: dict, count: int, batch_size: int, languages=None):
#     provider = cfg.get("provider", "gemini").lower()
#     if provider == "gemini" and not cfg.get("gemini_api_key"):
#         raise RuntimeError("Set GEMINI_API_KEY in .env to generate the corpus")
#     if provider == "ollama" and not cfg.get("ollama_model"):
#         raise RuntimeError("Set OLLAMA_MODEL in .env to generate the corpus")
#     if provider not in ("gemini", "ollama"):
#         raise RuntimeError(f"Corpus generation supports Gemini and Ollama; configured provider is '{provider}'.")
#     languages = languages or ["English", "Simplified Chinese"]
#     seeds = _read_json(output_dir / "scenario_seeds.json")
#     if not seeds:
#         raise RuntimeError("scenario_seeds.json is empty. Run ingest, extract, and build first.")
#     paths = {k: output_dir / f"{k}.jsonl" for k in ("scenario_blueprints", "conversations", "conversation_annotations")}
#     complete = {p.stem: {r.get("record_id") for r in _read_jsonl(p)} for p in paths.values()}
#     # Deterministic queue makes restarts resume the same planned records.
#     queue = []
#     for i in range(count):
#         seed = seeds[i % len(seeds)]
#         lang = languages[i % len(languages)]
#         rid = f"case_{i+1:08d}_{'en' if lang == 'English' else 'zh'}"
#         queue.append((rid, seed, lang, i))
#     for start in range(0, len(queue), batch_size):
#         batch = queue[start:start + batch_size]
#         model_name = cfg["gemini_model"] if provider == "gemini" else cfg["ollama_model"]
#         print(f"[corpus] provider={provider} model={model_name} | batch {start//batch_size+1}, records {start+1}-{start+len(batch)} of {count}", flush=True)
#         for rid, seed, lang, i in batch:
#             refs = seed.get("source_refs", [])
#             blueprint = next((r for r in _read_jsonl(paths["scenario_blueprints"]) if r.get("record_id") == rid), None)
#             if not blueprint:
#                 data = _request(_blueprint_prompt(seed, lang), cfg, BLUEPRINT_SCHEMA, rid + " blueprint")
#                 blueprint = {"record_id": rid, "language": lang, "modality": (seed.get("modality_relevance") or ["cross_modality"])[0],
#                              "blueprint": data, "source_refs": refs, "is_synthetic": True}
#                 _append(paths["scenario_blueprints"], blueprint)
#             conversation = next((r for r in _read_jsonl(paths["conversations"]) if r.get("record_id") == rid), None)
#             if not conversation:
#                 modality = blueprint["modality"]
#                 lengths = ["short", "medium", "long"]
#                 data = _request(_conversation_prompt(blueprint["blueprint"], lang, lengths[i % 3], modality), cfg, CONVERSATION_SCHEMA, rid + " dialogue")
#                 conversation = {"record_id": rid, "language": lang, "modality": modality, "sessions": data.get("sessions", []),
#                                 "source_refs": refs, "is_synthetic": True}
#                 _append(paths["conversations"], conversation)
#             annotated = next((r for r in _read_jsonl(paths["conversation_annotations"]) if r.get("record_id") == rid), None)
#             if not annotated:
#                 data = _request(_annotation_prompt(conversation, blueprint["blueprint"], lang), cfg, ANNOTATION_SCHEMA, rid + " annotations")
#                 annotated = {"record_id": rid, "language": lang, **data, "source_refs": refs, "is_synthetic": True}
#                 _append(paths["conversation_annotations"], annotated)
#             print(f"[corpus] {rid} complete ({lang})", flush=True)


# def validate_corpus(output_dir: Path):
#     blueprints = _read_jsonl(output_dir / "scenario_blueprints.jsonl")
#     conversations = _read_jsonl(output_dir / "conversations.jsonl")
#     annotations = _read_jsonl(output_dir / "conversation_annotations.jsonl")
#     errors = []
#     maps = [{r.get("record_id"): r for r in rows} for rows in (blueprints, conversations, annotations)]
#     for rid, bp in maps[0].items():
#         conv, ann = maps[1].get(rid), maps[2].get(rid)
#         if not bp.get("source_refs"): errors.append(f"{rid}: missing source refs")
#         if not conv: errors.append(f"{rid}: missing conversation")
#         elif len(conv.get("sessions", [])) < 2: errors.append(f"{rid}: fewer than two sessions")
#         elif any(not s.get("turns") for s in conv["sessions"]): errors.append(f"{rid}: session without turns")
#         if not ann: errors.append(f"{rid}: missing annotations")
#         elif not ann.get("turn_annotations"): errors.append(f"{rid}: no turn annotations")
#     ids = [r.get("record_id") for rows in (blueprints, conversations, annotations) for r in rows]
#     if len(ids) != len(set((id(x), x) for x in ids)):
#         pass
#     for issue in errors: print("ERROR:", issue)
#     print(f"Corpus validation: {len(errors)} error(s); {len(maps[1])} conversations")
#     return not errors


from pathlib import Path

script = r'''"""Checkpointed bilingual synthetic counselling corpus generation.

Supported providers:
    gemini, openai, openrouter, ollama

Provider/model selection is read from environment variables (loaded from .env when
python-dotenv is installed), with cfg values as fallbacks.

The scheduler is deliberately sequential. It enforces a rolling RPM/TPM budget,
keeps JSONL checkpoints, and persists long provider cooldowns so a process can
exit safely rather than sleeping for hours.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProviderCooldownError(RuntimeError):
    """A provider/model is in a persisted cooldown; resume after retry_at."""


class ProviderConfigurationError(RuntimeError):
    """Provider configuration is missing or invalid."""


class ProviderQuotaExhaustedError(RuntimeError):
    """A non-retryable billing/usage quota has been exhausted."""


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env without overriding variables already exported by the shell."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        # The main application may already have loaded .env. This is optional.
        pass


def _setting(cfg: dict, key: str, env_name: str | None = None,
             default: Any = None, env_first: bool = False) -> Any:
    """Read a setting from cfg/environment, ignoring empty strings."""
    env_name = env_name or key.upper()
    env_value = os.getenv(env_name)
    cfg_value = cfg.get(key) if isinstance(cfg, dict) else None

    if env_first:
        value = env_value if env_value not in (None, "") else cfg_value
    else:
        value = cfg_value if cfg_value not in (None, "") else env_value

    return default if value in (None, "") else value


def _int_setting(cfg: dict, key: str, env_name: str | None = None,
                 default: int = 0, minimum: int = 0) -> int:
    value = _setting(cfg, key, env_name, default)
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _float_setting(cfg: dict, key: str, env_name: str | None = None,
                   default: float = 0.0, minimum: float = 0.0) -> float:
    value = _setting(cfg, key, env_name, default)
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


def _provider_and_model(cfg: dict) -> tuple[str, str]:
    """Resolve the active provider/model; environment values take precedence."""
    provider = str(_setting(
        cfg, "provider", "LLM_PROVIDER", "gemini", env_first=True
    )).strip().lower()

    model_keys = {
        "openai": ("openai_model", "OPENAI_MODEL"),
        "openrouter": ("openrouter_model", "OPENROUTER_MODEL"),
        "gemini": ("gemini_model", "GEMINI_MODEL"),
        "ollama": ("ollama_model", "OLLAMA_MODEL"),
    }
    if provider not in model_keys:
        raise ProviderConfigurationError(
            f"Unsupported LLM_PROVIDER={provider!r}. "
            "Choose openai, openrouter, gemini, or ollama."
        )

    cfg_key, env_key = model_keys[provider]
    model = str(_setting(cfg, cfg_key, env_key, "")).strip()
    if not model:
        raise ProviderConfigurationError(
            f"{env_key} is empty for provider {provider!r}."
        )
    return provider, model


def _provider_display(cfg: dict) -> str:
    provider, model = _provider_and_model(cfg)
    return f"{provider.upper()} ({model})"


# ---------------------------------------------------------------------------
# JSONL/checkpoint helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# Cooldown persistence and API error classification
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _error_text(exc: Exception) -> str:
    parts = [str(exc), type(exc).__name__]
    for attr in ("code", "type", "message", "status_code"):
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(str(value))
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    return " ".join(parts).lower()


def _response_headers(exc: Exception):
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or {}


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read Retry-After and common textual retry durations."""
    headers = _response_headers(exc)
    retry_after = None
    try:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        pass

    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            # HTTP-date Retry-After
            try:
                from email.utils import parsedate_to_datetime
                target = parsedate_to_datetime(str(retry_after))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(0.0, (target - _utc_now()).total_seconds())
            except Exception:
                pass

    text = _error_text(exc)
    patterns = [
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s)\b", 1),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|m)\b", 60),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", 3600),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(days?|d)\b", 86400),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)) * multiplier
    return None


def _is_billing_exhaustion(exc: Exception) -> bool:
    text = _error_text(exc)
    markers = (
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "payment required",
        "billing",
        "out of credits",
    )
    return any(marker in text for marker in markers)


def _is_daily_quota_exhaustion(exc: Exception) -> bool:
    text = _error_text(exc)
    daily_markers = (
        "per day", "per-day", "daily quota", "daily limit",
        "requests per day", "tokens per day", "per_day",
        "quota exceeded for day", "daily requests",
    )
    return any(marker in text for marker in daily_markers)


def _is_transient_error(exc: Exception) -> bool:
    text = _error_text(exc)
    markers = (
        "429", "rate limit", "ratelimit", "resource_exhausted",
        "too many requests", "503", "502", "500",
        "server_is_overloaded", "temporarily unavailable",
        "timeout", "timed out", "connection reset",
        "connection error", "overloaded", "slow_down",
    )
    return any(marker in text for marker in markers)


def _cooldown_path(output_dir: Path) -> Path:
    return output_dir / "corpus_cooldown.json"


def _save_cooldown(output_dir: Path, provider: str, model: str,
                   retry_at: datetime, reason: str) -> None:
    path = _cooldown_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": provider,
        "model": model,
        "retry_at": _iso(retry_at),
        "reason": reason[:1500],
        "saved_at": _iso(_utc_now()),
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)


def _clear_cooldown(output_dir: Path) -> None:
    path = _cooldown_path(output_dir)
    if path.exists():
        path.unlink()


def _check_persisted_cooldown(output_dir: Path, provider: str, model: str) -> None:
    path = _cooldown_path(output_dir)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        print(f"[corpus] WARNING: ignoring unreadable cooldown file: {path}")
        return

    if data.get("provider") != provider or data.get("model") != model:
        print(
            "[corpus] Existing cooldown belongs to "
            f"{data.get('provider')} ({data.get('model')}); "
            f"current target is {provider} ({model})."
        )
        return

    retry_at = _parse_datetime(data.get("retry_at", ""))
    if retry_at is None or retry_at <= _utc_now():
        _clear_cooldown(output_dir)
        return

    remaining = int((retry_at - _utc_now()).total_seconds())
    raise ProviderCooldownError(
        f"{provider}/{model} is in cooldown until {_iso(retry_at)} "
        f"(about {remaining // 3600}h {(remaining % 3600) // 60}m remaining). "
        f"Reason: {data.get('reason', 'provider quota/rate limit')}. "
        "The completed JSONL records are checkpointed; rerun after cooldown."
    )


def _handle_provider_error(exc: Exception, cfg: dict, output_dir: Path,
                           provider: str, model: str, attempt: int) -> float | None:
    """Return a short retry delay, or raise for long/non-retryable cooldowns."""
    if _is_billing_exhaustion(exc):
        raise ProviderQuotaExhaustedError(
            f"{provider}/{model} API billing/usage quota is exhausted. "
            "This is not fixed by waiting: check API billing, credits, and "
            f"project spend limits. Original error: {exc}"
        ) from exc

    daily_hours = _float_setting(
        cfg, "quota_daily_cooldown_hours",
        "QUOTA_DAILY_COOLDOWN_HOURS", 12.0, 0.1
    )
    max_inline_wait = _float_setting(
        cfg, "max_inline_retry_wait",
        "MAX_INLINE_RETRY_WAIT", 900.0, 1.0
    )
    retry_after = _retry_after_seconds(exc)

    if _is_daily_quota_exhaustion(exc):
        delay = retry_after if retry_after is not None else daily_hours * 3600
        retry_at = _utc_now() + timedelta(seconds=delay)
        _save_cooldown(output_dir, provider, model, retry_at, str(exc))
        raise ProviderCooldownError(
            f"Daily quota appears exhausted for {provider}/{model}. "
            f"Cooldown saved until {_iso(retry_at)}. "
            "The process will stop safely; rerun after the cooldown."
        ) from exc

    if not _is_transient_error(exc):
        return None

    base = _float_setting(cfg, "quota_backoff_base",
                          "QUOTA_BACKOFF_BASE", 15.0, 0.1)
    maximum = _float_setting(cfg, "quota_backoff_max",
                             "QUOTA_BACKOFF_MAX", 300.0, 1.0)
    exponential = min(maximum, base * (2 ** max(0, attempt - 1)))
    delay = max(retry_after or 0.0, exponential)
    delay += random.uniform(0.0, min(5.0, delay * 0.15))

    if delay > max_inline_wait:
        retry_at = _utc_now() + timedelta(seconds=delay)
        _save_cooldown(output_dir, provider, model, retry_at, str(exc))
        raise ProviderCooldownError(
            f"{provider}/{model} requested a long retry delay "
            f"({delay / 60:.1f} minutes). Cooldown saved until "
            f"{_iso(retry_at)}. Rerun later; checkpoints are preserved."
        ) from exc

    return delay


# ---------------------------------------------------------------------------
# Adaptive sequential RPM/TPM scheduler
# ---------------------------------------------------------------------------

class _RateScheduler:
    """Conservative rolling 60-second request/token budget."""

    def __init__(self, cfg: dict, provider: str, model: str):
        prefix = provider.upper()
        self.rpm = _int_setting(
            cfg, "provider_rpm_limit", f"{prefix}_RPM_LIMIT", 0
        )
        self.tpm = _int_setting(
            cfg, "provider_tpm_limit", f"{prefix}_TPM_LIMIT", 0
        )
        self.min_interval = _float_setting(
            cfg, "provider_min_request_interval",
            f"{prefix}_MIN_REQUEST_INTERVAL", 0.0
        )

        # Backwards compatibility with the existing Gemini config.
        if provider == "gemini" and self.min_interval <= 0:
            self.min_interval = _float_setting(
                cfg, "gemini_min_request_interval",
                "GEMINI_MIN_REQUEST_INTERVAL", 0.0
            )

        self.max_output = _int_setting(
            cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1
        )
        self.events: deque[tuple[float, int]] = deque()
        self.last_request = 0.0
        self.provider = provider
        self.model = model

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Conservative approximation; use a provider tokenizer for exact counts
        # if one is available. CJK text can tokenize differently from English.
        return max(1, (len(text) + 2) // 3)

    def wait_turn(self, prompt: str) -> None:
        now = time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()

        estimated_input = self._estimate_tokens(prompt)
        # Many providers account against the larger of estimated prompt tokens
        # and the requested output cap. This is a scheduling reservation, not
        # a claim about exact provider-side accounting.
        reservation = max(estimated_input, self.max_output)

        waits = []
        if self.min_interval > 0 and self.last_request:
            waits.append(self.min_interval - (now - self.last_request))

        if self.rpm > 0 and len(self.events) >= self.rpm:
            waits.append(60 - (now - self.events[0][0]))

        used_tokens = sum(tokens for _, tokens in self.events)
        if self.tpm > 0 and used_tokens + reservation > self.tpm and self.events:
            remaining_events = list(self.events)
            cumulative = used_tokens
            token_wait = 0.0
            for timestamp, tokens in remaining_events:
                cumulative -= tokens
                token_wait = max(0.0, 60 - (now - timestamp))
                if cumulative + reservation <= self.tpm:
                    break
            else:
                token_wait = max(token_wait, 60 - (now - remaining_events[0][0]))
            waits.append(token_wait)

        wait_seconds = max(waits, default=0.0)
        if wait_seconds > 0:
            print(
                f"[corpus] scheduler: waiting {wait_seconds:.1f}s "
                f"(RPM={self.rpm or 'unset'}, TPM={self.tpm or 'unset'}, "
                f"reserved≈{reservation} tokens)",
                flush=True,
            )
            time.sleep(wait_seconds + 0.05)

        now = time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
        self.events.append((now, reservation))
        self.last_request = now

        if self.tpm > 0 and reservation > self.tpm:
            print(
                f"[corpus] WARNING: one request reserves ≈{reservation} tokens, "
                f"above configured TPM={self.tpm}; reduce MAX_OUTPUT_TOKENS "
                "or split the task.",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def _call_openai_compatible(prompt: str, cfg: dict, model: str,
                            provider: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK missing. Install with: pip install openai"
        ) from exc

    if provider == "openai":
        api_key = _setting(cfg, "openai_api_key", "OPENAI_API_KEY", "")
        base_url = None
    else:
        api_key = _setting(cfg, "openrouter_api_key", "OPENROUTER_API_KEY", "")
        base_url = _setting(
            cfg, "openrouter_base_url", "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"
        )

    if not api_key:
        env_name = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
        raise ProviderConfigurationError(f"Set {env_name} in .env.")

    timeout = _float_setting(
        cfg, "request_timeout", "REQUEST_TIMEOUT", 600.0, 1.0
    )
    client_args = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": 0,  # retries are handled by this module
    }
    if base_url:
        client_args["base_url"] = base_url
    client = OpenAI(**client_args)

    max_output = _int_setting(
        cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1
    )
    temperature = _float_setting(cfg, "temperature", "TEMPERATURE", 0.1)

    # JSON mode is broadly compatible with OpenAI and OpenRouter models.
    # The requested schema is included in the prompt by _request().
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return exactly one valid JSON object. Do not wrap it in "
                    "Markdown fences. Follow the JSON schema included in the "
                    "user message."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_completion_tokens=max_output,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"{provider} returned an empty response")
    return json.loads(content)


def _call_gemini(prompt: str, cfg: dict, model: str, schema: dict) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK missing. Install with: pip install google-genai"
        ) from exc

    api_key = _setting(cfg, "gemini_api_key", "GEMINI_API_KEY", "")
    if not api_key:
        raise ProviderConfigurationError("Set GEMINI_API_KEY in .env.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=_float_setting(cfg, "temperature", "TEMPERATURE", 0.1),
            max_output_tokens=_int_setting(
                cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1
            ),
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")
    return json.loads(response.text)


def _call_ollama(prompt: str, cfg: dict, model: str) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests missing. Install with: pip install requests") from exc

    url = str(_setting(
        cfg, "ollama_url", "OLLAMA_URL", "http://localhost:11434"
    )).strip().rstrip("/")
    response = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": _float_setting(
                    cfg, "temperature", "TEMPERATURE", 0.1
                ),
                "num_predict": _int_setting(
                    cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1
                ),
            },
        },
        timeout=_float_setting(
            cfg, "request_timeout", "REQUEST_TIMEOUT", 600.0, 1.0
        ),
    )
    response.raise_for_status()
    raw = response.json().get("response", "")
    if not raw:
        raise RuntimeError("Ollama returned an empty response")
    return json.loads(raw)


def _call_model(prompt: str, cfg: dict, schema: dict) -> dict:
    provider, model = _provider_and_model(cfg)

    if provider in ("openai", "openrouter"):
        return _call_openai_compatible(prompt, cfg, model, provider)
    if provider == "gemini":
        return _call_gemini(prompt, cfg, model, schema)
    if provider == "ollama":
        return _call_ollama(prompt, cfg, model)

    raise ProviderConfigurationError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Request wrapper: schema prompt, rate scheduling, retry/cooldown handling
# ---------------------------------------------------------------------------

def _request(prompt: str, cfg: dict, schema: dict, label: str,
             output_dir: Path, scheduler: _RateScheduler) -> dict:
    provider, model = _provider_and_model(cfg)
    retries = _int_setting(cfg, "corpus_retries", "CORPUS_RETRIES", 5, 1)

    schema_instruction = (
        "\n\nReturn a JSON object conforming to this schema. "
        "Do not include Markdown fences or explanatory text:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    full_prompt = prompt + schema_instruction

    for attempt in range(1, retries + 1):
        scheduler.wait_turn(full_prompt)
        try:
            print(
                f"  {label}: {provider.upper()} / {model} "
                f"request {attempt}/{retries}",
                flush=True,
            )
            result = _call_model(full_prompt, cfg, schema)
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"{provider}/{model} returned JSON type "
                    f"{type(result).__name__}; expected an object."
                )
            return result
        except ProviderConfigurationError:
            raise
        except ProviderQuotaExhaustedError:
            raise
        except ProviderCooldownError:
            raise
        except Exception as exc:
            if _is_billing_exhaustion(exc):
                _handle_provider_error(
                    exc, cfg, output_dir, provider, model, attempt
                )
            if _is_daily_quota_exhaustion(exc):
                _handle_provider_error(
                    exc, cfg, output_dir, provider, model, attempt
                )

            if attempt >= retries:
                # Long Retry-After values must persist a cooldown even on the
                # final attempt. Short transient errors are reported normally.
                _handle_provider_error(
                    exc, cfg, output_dir, provider, model, attempt
                )
                raise RuntimeError(
                    f"{provider}/{model} failed after {retries} attempts "
                    f"for {label}: {exc}"
                ) from exc

            delay = _handle_provider_error(
                exc, cfg, output_dir, provider, model, attempt
            )
            if delay is None:
                raise RuntimeError(
                    f"Non-retryable {provider}/{model} error for {label}: {exc}"
                ) from exc

            print(
                f"  [{provider}/{model}] request failed "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Request loop exited unexpectedly")


# ---------------------------------------------------------------------------
# Generation prompts and schemas
# ---------------------------------------------------------------------------

BLUEPRINT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "client": {"type": "OBJECT"},
        "presenting_problem": {"type": "STRING"},
        "beliefs": {"type": "ARRAY", "items": {"type": "STRING"}},
        "emotions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "therapy_frame": {"type": "STRING"},
        "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "title", "client", "presenting_problem", "beliefs", "emotions",
        "therapy_frame", "source_refs"
    ],
}

CONVERSATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sessions": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["sessions", "source_refs"],
}

ANNOTATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "turn_annotations": {
            "type": "ARRAY", "items": {"type": "OBJECT"}
        },
        "session_summaries": {
            "type": "ARRAY", "items": {"type": "OBJECT"}
        },
    },
    "required": ["turn_annotations", "session_summaries"],
}


def _blueprint_prompt(seed: dict, language: str) -> str:
    return (
        "Create one original fictional counselling case blueprint in " + language +
        ". Return JSON matching the schema. Client profile must be internally "
        "consistent and include nuanced history, strengths, context, multiple "
        "competing beliefs, emotions and interpersonal dynamics. Use source only "
        "as a high-level therapy concept anchor; do not reproduce book text or "
        "imply the fictional client appears in a book. Mark uncertainty. Maintain "
        "safe, non-diagnostic framing.\n" +
        json.dumps(seed, ensure_ascii=False)
    )


def _conversation_prompt(blueprint: dict, language: str,
                         length: str, modality: str) -> str:
    return (
        f"Write a realistic fictional multi-session counselling conversation in "
        f"{language}. Modality: {modality}; length profile: {length}. Return JSON "
        "with sessions, each having session_number, goal, turns array of "
        "{turn_id, speaker(client|therapist), text, emotional_state, belief_state}, "
        "and ending_state. Include 3-6 sessions, varied turn counts, client agency, "
        "ruptures/misunderstandings, unresolved material, and intervention "
        "consequences. CBT uses collaborative formulation and careful experiments; "
        "person-centered uses empathic reflection and non-directivity. Do not claim "
        "clinical efficacy, avoid diagnosing, handle risk disclosures by "
        "prioritizing immediate human safety assessment. Keep dialogue original, "
        "not copied from source.\n" +
        json.dumps(blueprint, ensure_ascii=False)
    )


def _annotation_prompt(conversation: dict, blueprint: dict,
                       language: str) -> str:
    return (
        f"Annotate this fictional counselling corpus in {language}. Return full "
        "inference annotations: for every turn record known facts, hypotheses "
        "(at least two competing interpretations where plausible), supporting "
        "evidence turn IDs, contradictions, confidence 0..1, uncertainty, client "
        "beliefs, therapist decision and rationale, intervention consequence, "
        "emotional change, safety flags. Distinguish stated, inferred, uncertain, "
        "contradicted. Never present hypotheses as facts. Add session summaries.\n" +
        json.dumps(
            {"blueprint": blueprint, "conversation": conversation},
            ensure_ascii=False
        )
    )


# ---------------------------------------------------------------------------
# Main corpus generation
# ---------------------------------------------------------------------------

def generate_corpus(output_dir: Path, cfg: dict, count: int,
                    batch_size: int, languages=None):
    """Generate/resume synthetic cases, preserving each completed stage."""
    _load_dotenv()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider, model = _provider_and_model(cfg)
    if provider == "gemini" and not _setting(
        cfg, "gemini_api_key", "GEMINI_API_KEY", ""
    ):
        raise ProviderConfigurationError("Set GEMINI_API_KEY in .env.")
    if provider == "openai" and not _setting(
        cfg, "openai_api_key", "OPENAI_API_KEY", ""
    ):
        raise ProviderConfigurationError("Set OPENAI_API_KEY in .env.")
    if provider == "openrouter" and not _setting(
        cfg, "openrouter_api_key", "OPENROUTER_API_KEY", ""
    ):
        raise ProviderConfigurationError("Set OPENROUTER_API_KEY in .env.")

    _check_persisted_cooldown(output_dir, provider, model)

    if count < 0:
        raise ValueError("count must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    languages = languages or ["English", "Simplified Chinese"]
    seeds = _read_json(output_dir / "scenario_seeds.json")
    if not seeds:
        raise RuntimeError(
            "scenario_seeds.json is empty. Run ingest, extract, and build first."
        )

    paths = {
        key: output_dir / f"{key}.jsonl"
        for key in (
            "scenario_blueprints",
            "conversations",
            "conversation_annotations",
        )
    }

    # Index checkpoints once instead of rereading every JSONL file per case.
    records = {key: _read_jsonl(path) for key, path in paths.items()}
    indexes = {
        key: {row.get("record_id"): row for row in rows}
        for key, rows in records.items()
    }

    queue = []
    for i in range(count):
        seed = seeds[i % len(seeds)]
        language = languages[i % len(languages)]
        record_id = f"case_{i + 1:08d}_{'en' if language == 'English' else 'zh'}"
        queue.append((record_id, seed, language, i))

    scheduler = _RateScheduler(cfg, provider, model)
    print(
        f"[corpus] provider={provider} model={model} "
        f"| queued={len(queue)} | batch_size={batch_size}",
        flush=True,
    )

    for start in range(0, len(queue), batch_size):
        batch = queue[start:start + batch_size]
        print(
            f"[corpus] batch {start // batch_size + 1}, "
            f"records {start + 1}-{start + len(batch)} of {count}",
            flush=True,
        )

        for record_id, seed, language, i in batch:
            refs = seed.get("source_refs", [])

            blueprint_row = indexes["scenario_blueprints"].get(record_id)
            if not blueprint_row:
                data = _request(
                    _blueprint_prompt(seed, language), cfg,
                    BLUEPRINT_SCHEMA, record_id + " blueprint",
                    output_dir, scheduler,
                )
                blueprint_row = {
                    "record_id": record_id,
                    "language": language,
                    "modality": (
                        seed.get("modality_relevance") or ["cross_modality"]
                    )[0],
                    "blueprint": data,
                    "source_refs": refs,
                    "is_synthetic": True,
                    "generation_provider": provider,
                    "generation_model": model,
                }
                _append(paths["scenario_blueprints"], blueprint_row)
                indexes["scenario_blueprints"][record_id] = blueprint_row

            conversation_row = indexes["conversations"].get(record_id)
            if not conversation_row:
                modality = blueprint_row["modality"]
                lengths = ["short", "medium", "long"]
                data = _request(
                    _conversation_prompt(
                        blueprint_row["blueprint"], language,
                        lengths[i % len(lengths)], modality
                    ),
                    cfg, CONVERSATION_SCHEMA, record_id + " dialogue",
                    output_dir, scheduler,
                )
                conversation_row = {
                    "record_id": record_id,
                    "language": language,
                    "modality": modality,
                    "sessions": data.get("sessions", []),
                    "source_refs": refs,
                    "is_synthetic": True,
                    "generation_provider": provider,
                    "generation_model": model,
                }
                _append(paths["conversations"], conversation_row)
                indexes["conversations"][record_id] = conversation_row

            annotation_row = indexes["conversation_annotations"].get(record_id)
            if not annotation_row:
                data = _request(
                    _annotation_prompt(
                        conversation_row, blueprint_row["blueprint"], language
                    ),
                    cfg, ANNOTATION_SCHEMA, record_id + " annotations",
                    output_dir, scheduler,
                )
                annotation_row = {
                    "record_id": record_id,
                    "language": language,
                    **data,
                    "source_refs": refs,
                    "is_synthetic": True,
                    "generation_provider": provider,
                    "generation_model": model,
                }
                _append(paths["conversation_annotations"], annotation_row)
                indexes["conversation_annotations"][record_id] = annotation_row

            print(
                f"[corpus] {record_id} complete ({language}) "
                f"via {provider}/{model}",
                flush=True,
            )

    print(
        f"[corpus] generation finished. "
        f"blueprints={len(indexes['scenario_blueprints'])}, "
        f"conversations={len(indexes['conversations'])}, "
        f"annotations={len(indexes['conversation_annotations'])}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_corpus(output_dir: Path):
    blueprints = _read_jsonl(Path(output_dir) / "scenario_blueprints.jsonl")
    conversations = _read_jsonl(Path(output_dir) / "conversations.jsonl")
    annotations = _read_jsonl(Path(output_dir) / "conversation_annotations.jsonl")

    errors = []
    maps = [
        {row.get("record_id"): row for row in rows}
        for rows in (blueprints, conversations, annotations)
    ]

    for record_id, blueprint in maps[0].items():
        conversation = maps[1].get(record_id)
        annotation = maps[2].get(record_id)

        if not blueprint.get("source_refs"):
            errors.append(f"{record_id}: missing source refs")

        if not conversation:
            errors.append(f"{record_id}: missing conversation")
        else:
            sessions = conversation.get("sessions", [])
            if len(sessions) < 2:
                errors.append(f"{record_id}: fewer than two sessions")
            for session_index, session in enumerate(sessions, start=1):
                if not session.get("turns"):
                    errors.append(
                        f"{record_id}: session {session_index} without turns"
                    )

        if not annotation:
            errors.append(f"{record_id}: missing annotations")
        elif not annotation.get("turn_annotations"):
            errors.append(f"{record_id}: no turn annotations")

    # Detect duplicate IDs within each artifact independently.
    for name, rows in (
        ("blueprints", blueprints),
        ("conversations", conversations),
        ("annotations", annotations),
    ):
        ids = [row.get("record_id") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}: duplicate record_id values")

    for issue in errors:
        print("ERROR:", issue)

    print(
        f"Corpus validation: {len(errors)} error(s); "
        f"{len(maps[1])} conversations"
    )
    return not errors
'''

out = Path("/mnt/data/corpus.py")
out.write_text(script, encoding="utf-8")
print(f"Created {out} ({len(script.splitlines())} lines, {out.stat().st_size:,} bytes)")

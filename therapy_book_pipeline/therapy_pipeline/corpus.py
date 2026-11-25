"""Checkpointed bilingual synthetic counselling corpus generation.

Providers:
  Gemini, Ollama, OpenAI, OpenRouter, NVIDIA NIM, Hugging Face,
  Groq, Together, Fireworks, DeepSeek, Mistral, Cerebras, xAI,
  and arbitrary OpenAI-compatible endpoints.

Run behaviour:
  CORPUS_RUN_MODE=resume (default):
      Continue the corpus in OUTPUT_DIR. --count is the TOTAL target.
      Example: an existing 100-record corpus + --count 250 generates 101..250.
  CORPUS_RUN_MODE=new:
      Create an isolated run under OUTPUT_DIR/corpus_runs/<run-id>/.
      Existing corpus files are not modified.

Each stage is checkpointed independently in JSONL and fsynced:
  scenario_blueprints.jsonl
  conversations.jsonl
  annotation_parts.jsonl       (one checkpoint per session)
  conversation_annotations.jsonl (merged final annotations)

A rerun skips completed stages and completed annotation sessions. Invalid model
JSON is retried with a repair instruction; failed raw responses are preserved
under corpus_failures/. Provider cooldowns are persisted per run directory.

Generation is sequential. batch_size controls progress grouping only; it does
not increase API concurrency or reduce the number of requests.
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


class ProviderCooldownError(RuntimeError):
    """Provider/model is in a persisted cooldown; rerun after retry_at."""


class ProviderConfigurationError(RuntimeError):
    """Provider configuration is missing or invalid."""


class ProviderQuotaExhaustedError(RuntimeError):
    """A non-retryable billing/usage quota has been exhausted."""


class ModelJSONError(RuntimeError):
    """A model returned content that could not be parsed as one JSON object."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


def _setting(cfg: dict, key: str, env_name: str | None = None,
             default: Any = None, env_first: bool = False) -> Any:
    env_value = os.getenv(env_name or key.upper())
    cfg_value = cfg.get(key) if isinstance(cfg, dict) else None
    value = (env_value if env_value not in (None, "") else cfg_value) if env_first else (
        cfg_value if cfg_value not in (None, "") else env_value
    )
    return default if value in (None, "") else value


def _bool_setting(cfg: dict, key: str, env_name: str, default: bool) -> bool:
    value = str(_setting(cfg, key, env_name, str(default).lower())).strip().lower()
    return value in ("1", "true", "yes", "on")


def _int_setting(cfg: dict, key: str, env_name: str | None = None,
                 default: int = 0, minimum: int = 0) -> int:
    try:
        return max(minimum, int(_setting(cfg, key, env_name, default)))
    except (TypeError, ValueError):
        return default


def _float_setting(cfg: dict, key: str, env_name: str | None = None,
                   default: float = 0.0, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(_setting(cfg, key, env_name, default)))
    except (TypeError, ValueError):
        return default


_OPENAI_COMPATIBLE = {
    "openai": {
        "key": "OPENAI_API_KEY", "model": "OPENAI_MODEL", "base": None,
        "cfg_key": "openai_api_key", "model_cfg": "openai_model",
    },
    "openrouter": {
        "key": "OPENROUTER_API_KEY", "model": "OPENROUTER_MODEL",
        "base": "https://openrouter.ai/api/v1",
        "cfg_key": "openrouter_api_key", "model_cfg": "openrouter_model",
        "base_env": "OPENROUTER_BASE_URL", "base_cfg": "openrouter_base_url",
    },
    "nvidia": {
        "key": "NVIDIA_API_KEY", "model": "NVIDIA_MODEL",
        "base": "https://integrate.api.nvidia.com/v1",
        "cfg_key": "nvidia_api_key", "model_cfg": "nvidia_model",
        "base_env": "NVIDIA_BASE_URL", "base_cfg": "nvidia_base_url",
    },
    "huggingface": {
        "key": "HUGGINGFACE_API_KEY", "model": "HUGGINGFACE_MODEL",
        "base": "https://router.huggingface.co/v1",
        "cfg_key": "huggingface_api_key", "model_cfg": "huggingface_model",
        "base_env": "HUGGINGFACE_BASE_URL", "base_cfg": "huggingface_base_url",
    },
    "groq": {
        "key": "GROQ_API_KEY", "model": "GROQ_MODEL",
        "base": "https://api.groq.com/openai/v1",
        "cfg_key": "groq_api_key", "model_cfg": "groq_model",
        "base_env": "GROQ_BASE_URL", "base_cfg": "groq_base_url",
    },
    "together": {
        "key": "TOGETHER_API_KEY", "model": "TOGETHER_MODEL",
        "base": "https://api.together.xyz/v1",
        "cfg_key": "together_api_key", "model_cfg": "together_model",
        "base_env": "TOGETHER_BASE_URL", "base_cfg": "together_base_url",
    },
    "fireworks": {
        "key": "FIREWORKS_API_KEY", "model": "FIREWORKS_MODEL",
        "base": "https://api.fireworks.ai/inference/v1",
        "cfg_key": "fireworks_api_key", "model_cfg": "fireworks_model",
        "base_env": "FIREWORKS_BASE_URL", "base_cfg": "fireworks_base_url",
    },
    "deepseek": {
        "key": "DEEPSEEK_API_KEY", "model": "DEEPSEEK_MODEL",
        "base": "https://api.deepseek.com/v1",
        "cfg_key": "deepseek_api_key", "model_cfg": "deepseek_model",
        "base_env": "DEEPSEEK_BASE_URL", "base_cfg": "deepseek_base_url",
    },
    "mistral": {
        "key": "MISTRAL_API_KEY", "model": "MISTRAL_MODEL",
        "base": "https://api.mistral.ai/v1",
        "cfg_key": "mistral_api_key", "model_cfg": "mistral_model",
        "base_env": "MISTRAL_BASE_URL", "base_cfg": "mistral_base_url",
    },
    "cerebras": {
        "key": "CEREBRAS_API_KEY", "model": "CEREBRAS_MODEL",
        "base": "https://api.cerebras.ai/v1",
        "cfg_key": "cerebras_api_key", "model_cfg": "cerebras_model",
        "base_env": "CEREBRAS_BASE_URL", "base_cfg": "cerebras_base_url",
    },
    "xai": {
        "key": "XAI_API_KEY", "model": "XAI_MODEL",
        "base": "https://api.x.ai/v1",
        "cfg_key": "xai_api_key", "model_cfg": "xai_model",
        "base_env": "XAI_BASE_URL", "base_cfg": "xai_base_url",
    },
    "custom": {
        "key": "LLM_API_KEY", "model": "LLM_MODEL", "base": None,
        "cfg_key": "llm_api_key", "model_cfg": "llm_model",
        "base_env": "LLM_BASE_URL", "base_cfg": "llm_base_url",
    },
}


def _provider_and_model(cfg: dict) -> tuple[str, str]:
    provider = str(_setting(cfg, "provider", "LLM_PROVIDER", "gemini", True)).strip().lower()
    provider = {"hf": "huggingface", "hugging_face": "huggingface",
                "nvidia_nim": "nvidia"}.get(provider, provider)
    if provider == "gemini":
        model = _setting(cfg, "gemini_model", "GEMINI_MODEL", "")
    elif provider == "ollama":
        model = _setting(cfg, "ollama_model", "OLLAMA_MODEL", "")
    elif provider in _OPENAI_COMPATIBLE:
        spec = _OPENAI_COMPATIBLE[provider]
        model = _setting(cfg, spec["model_cfg"], spec["model"], "")
    else:
        raise ProviderConfigurationError(
            f"Unsupported LLM_PROVIDER={provider!r}. Supported: gemini, ollama, "
            + ", ".join(_OPENAI_COMPATIBLE)
        )
    model = str(model).strip()
    if not model:
        raise ProviderConfigurationError(f"Model is not configured for provider {provider!r}.")
    return provider, model


# ---------------------------------------------------------------------------
# JSON / JSONL persistence
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}. "
                    "Do not delete the file; inspect or restore the damaged line first."
                ) from exc
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _index_rows(rows: list[dict], key: str = "record_id") -> dict:
    # Last valid checkpoint wins if a user manually restored/replayed a row.
    result = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            result[value] = row
    return result


def _valid_blueprint(row: dict) -> bool:
    bp = row.get("blueprint") or {}
    client = bp.get("client") or {}
    return bool(row.get("source_refs") and bp.get("title") and
        bp.get("presenting_problem") and isinstance(bp.get("beliefs"), list) and
        len(bp["beliefs"]) >= 2 and isinstance(bp.get("emotions"), list) and
        bp.get("therapy_frame") and client.get("pseudonym") and
        isinstance(client.get("age"), int) and client.get("occupation_or_role") and
        client.get("cultural_and_family_context") and
        isinstance(client.get("relevant_history"), list) and client["relevant_history"] and
        isinstance(client.get("strengths_and_supports"), list) and client["strengths_and_supports"])


def _valid_conversation(row: dict) -> bool:
    sessions = row.get("sessions") or []
    return (len(sessions) >= 3 and all(
        isinstance(s.get("turns"), list) and len(s["turns"]) >= 8 and
        all(t.get("speaker") in ("client", "therapist") and t.get("text")
            for t in s["turns"])
        for s in sessions
    ))


# ---------------------------------------------------------------------------
# Cooldown persistence and error classification
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
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(str(response.text))
        except Exception:
            pass
    return " ".join(parts).lower()


def _response_headers(exc: Exception):
    return getattr(getattr(exc, "response", None), "headers", None) or {}


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = _response_headers(exc)
    value = None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        pass
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                from email.utils import parsedate_to_datetime
                target = parsedate_to_datetime(str(value))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(0.0, (target - _utc_now()).total_seconds())
            except Exception:
                pass
    text = _error_text(exc)
    patterns = (
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s)\b", 1),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|m)\b", 60),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", 3600),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(days?|d)\b", 86400),
    )
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)) * multiplier
    return None


def _is_billing_exhaustion(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(x in text for x in (
        "insufficient_quota", "credit_balance_exhausted", "billing_hard_limit",
        "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
        "payment required", "billing", "out of credits",
    ))


def _is_daily_quota_exhaustion(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(x in text for x in (
        "per day", "per-day", "daily quota", "daily limit", "requests per day",
        "tokens per day", "per_day", "quota exceeded for day", "requestsperday",
    ))


def _is_transient_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(x in text for x in (
        "429", "rate limit", "ratelimit", "resource_exhausted", "too many requests",
        "503", "502", "500", "server_is_overloaded", "temporarily unavailable",
        "timeout", "timed out", "connection reset", "connection error", "overloaded",
        "slow_down", "service unavailable", "bad gateway", "internal server error",
    ))


def _cooldown_path(output_dir: Path) -> Path:
    return output_dir / "corpus_cooldown.json"


def _save_cooldown(output_dir: Path, provider: str, model: str,
                   retry_at: datetime, reason: str) -> None:
    _write_json_atomic(_cooldown_path(output_dir), {
        "provider": provider, "model": model, "retry_at": _iso(retry_at),
        "reason": reason[:1500], "saved_at": _iso(_utc_now()),
    })


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
        print("[corpus] Existing cooldown belongs to another provider/model; ignoring it.")
        return
    retry_at = _parse_datetime(data.get("retry_at", ""))
    if retry_at is None or retry_at <= _utc_now():
        _clear_cooldown(output_dir)
        return
    remaining = int((retry_at - _utc_now()).total_seconds())
    raise ProviderCooldownError(
        f"{provider}/{model} is in cooldown until {_iso(retry_at)} "
        f"(about {remaining // 3600}h {(remaining % 3600) // 60}m remaining). "
        "Checkpoints are preserved; rerun after the cooldown."
    )


def _handle_provider_error(exc: Exception, cfg: dict, output_dir: Path,
                           provider: str, model: str, attempt: int) -> float | None:
    if _is_billing_exhaustion(exc):
        raise ProviderQuotaExhaustedError(
            f"{provider}/{model} billing/credit quota exhausted. Check provider billing. "
            f"Original error: {exc}"
        ) from exc
    daily_hours = _float_setting(cfg, "quota_daily_cooldown_hours",
                                 "QUOTA_DAILY_COOLDOWN_HOURS", 12.0, 0.1)
    max_inline_wait = _float_setting(cfg, "max_inline_retry_wait",
                                     "MAX_INLINE_RETRY_WAIT", 900.0, 1.0)
    retry_after = _retry_after_seconds(exc)
    if _is_daily_quota_exhaustion(exc):
        delay = retry_after if retry_after is not None else daily_hours * 3600
        retry_at = _utc_now() + timedelta(seconds=delay)
        _save_cooldown(output_dir, provider, model, retry_at, str(exc))
        raise ProviderCooldownError(
            f"Daily quota appears exhausted for {provider}/{model}. "
            f"Cooldown saved until {_iso(retry_at)}; checkpoints are preserved."
        ) from exc
    if not _is_transient_error(exc):
        return None
    base = _float_setting(cfg, "quota_backoff_base", "QUOTA_BACKOFF_BASE", 15.0, 0.1)
    maximum = _float_setting(cfg, "quota_backoff_max", "QUOTA_BACKOFF_MAX", 300.0, 1.0)
    delay = max(retry_after or 0.0, min(maximum, base * (2 ** max(0, attempt - 1))))
    delay += random.uniform(0.0, min(10.0, delay * 0.20))
    if delay > max_inline_wait:
        retry_at = _utc_now() + timedelta(seconds=delay)
        _save_cooldown(output_dir, provider, model, retry_at, str(exc))
        raise ProviderCooldownError(
            f"{provider}/{model} requested a long retry delay ({delay / 60:.1f} minutes). "
            f"Cooldown saved until {_iso(retry_at)}; rerun later."
        ) from exc
    return delay


# ---------------------------------------------------------------------------
# Sequential RPM/TPM scheduler
# ---------------------------------------------------------------------------

class _RateScheduler:
    def __init__(self, cfg: dict, provider: str, model: str):
        prefix = provider.upper()
        self.rpm = _int_setting(cfg, "provider_rpm_limit", f"{prefix}_RPM_LIMIT", 0)
        self.tpm = _int_setting(cfg, "provider_tpm_limit", f"{prefix}_TPM_LIMIT", 0)
        self.min_interval = _float_setting(
            cfg, "provider_min_request_interval", f"{prefix}_MIN_REQUEST_INTERVAL", 0.0
        )
        if provider == "gemini" and self.min_interval <= 0:
            self.min_interval = _float_setting(
                cfg, "gemini_min_request_interval", "GEMINI_MIN_REQUEST_INTERVAL", 0.0
            )
        self.max_output = _int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1)
        self.events: deque[tuple[float, int]] = deque()
        self.last_request = 0.0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 2) // 3)

    def wait_turn(self, prompt: str, output_reservation: int | None = None) -> None:
        now = time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
        reservation = max(self._estimate_tokens(prompt), output_reservation or self.max_output)
        waits = []
        if self.min_interval > 0 and self.last_request:
            waits.append(self.min_interval - (now - self.last_request))
        if self.rpm > 0 and len(self.events) >= self.rpm:
            waits.append(60 - (now - self.events[0][0]))
        used = sum(tokens for _, tokens in self.events)
        if self.tpm > 0 and used + reservation > self.tpm and self.events:
            cumulative, token_wait = used, 0.0
            for timestamp, tokens in list(self.events):
                cumulative -= tokens
                token_wait = max(0.0, 60 - (now - timestamp))
                if cumulative + reservation <= self.tpm:
                    break
            waits.append(token_wait)
        wait_seconds = max(waits, default=0.0)
        if wait_seconds > 0:
            print(
                f"[corpus] scheduler: waiting {wait_seconds:.1f}s "
                f"(RPM={self.rpm or 'unset'}, TPM={self.tpm or 'unset'}, "
                f"reserved≈{reservation})", flush=True
            )
            time.sleep(wait_seconds + 0.05)
        now = time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
        self.events.append((now, reservation))
        self.last_request = now
        if self.tpm > 0 and reservation > self.tpm:
            print(
                f"[corpus] WARNING: request reservation ≈{reservation} exceeds "
                f"configured TPM={self.tpm}; reduce output tokens or split the task.",
                flush=True
            )


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def _compatible_credentials(cfg: dict, provider: str):
    spec = _OPENAI_COMPATIBLE[provider]
    api_key = _setting(cfg, spec["cfg_key"], spec["key"], "")
    base_url = spec.get("base")
    if spec.get("base_env"):
        base_url = _setting(cfg, spec["base_cfg"], spec["base_env"], base_url)
    if provider == "custom" and not base_url:
        raise ProviderConfigurationError("Set LLM_BASE_URL for the custom provider.")
    if not api_key and provider != "custom":
        raise ProviderConfigurationError(f"Set {spec['key']} in .env for provider {provider}.")
    return api_key or "not-needed", str(base_url).rstrip("/") if base_url else None


def _parse_model_json(raw: str, provider: str, diagnostic: str = "") -> dict:
    """Parse one JSON object; never silently accept partial/truncated JSON."""
    text = (raw or "").strip().lstrip("\ufeff")
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start and (start != 0 or end != len(text) - 1):
        candidates.append(text[start:end + 1])
    last_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ModelJSONError(
                    f"{provider} returned JSON type {type(parsed).__name__}; expected object.", raw
                )
            return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    detail = f"{provider} returned invalid JSON: {last_error}"
    if diagnostic:
        detail += f" | response diagnostic: {diagnostic}"
    raise ModelJSONError(detail, raw)


def _call_openai_compatible(prompt: str, cfg: dict, model: str, provider: str) -> tuple[dict, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK missing. Install with: pip install openai") from exc
    api_key, base_url = _compatible_credentials(cfg, provider)
    timeout = _float_setting(cfg, "request_timeout", "REQUEST_TIMEOUT", 600.0, 1.0)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    max_output = _int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1)
    request_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Return exactly one valid JSON object. No Markdown fences or commentary. "
                "Follow the schema in the user message."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output,
    }
    if _bool_setting(cfg, "llm_json_mode", "LLM_JSON_MODE", True):
        request_args["response_format"] = {"type": "json_object"}
    if _bool_setting(cfg, "llm_send_temperature", "LLM_SEND_TEMPERATURE", True):
        request_args["temperature"] = _float_setting(cfg, "temperature", "TEMPERATURE", 0.1)
    effort = _setting(cfg, "llm_reasoning_effort", "LLM_REASONING_EFFORT", "")
    if effort:
        request_args["reasoning_effort"] = str(effort)
    response = client.chat.completions.create(**request_args)
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if not content:
        raise RuntimeError(f"{provider} returned an empty response")
    finish = getattr(choice, "finish_reason", None)
    diagnostic = f"finish_reason={finish}" if finish else ""
    return _parse_model_json(content, provider, diagnostic), content


def _call_gemini(prompt: str, cfg: dict, model: str, schema: dict) -> tuple[dict, str]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini SDK missing. Install with: pip install google-genai") from exc
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
            max_output_tokens=_int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1),
        ),
    )
    raw = response.text or ""
    if not raw:
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        raise RuntimeError(f"Gemini returned an empty response (finish_reason={finish})")
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    usage = getattr(response, "usage_metadata", None)
    diagnostic = f"finish_reason={finish}; usage={usage}" if finish or usage else ""
    return _parse_model_json(raw, "gemini", diagnostic), raw


def _call_ollama(prompt: str, cfg: dict, model: str) -> tuple[dict, str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests missing. Install with: pip install requests") from exc
    url = str(_setting(cfg, "ollama_url", "OLLAMA_URL",
                       "http://localhost:11434")).strip().rstrip("/")
    response = requests.post(
        f"{url}/api/generate",
        json={
            "model": model, "prompt": prompt, "stream": False, "format": "json",
            "options": {
                "temperature": _float_setting(cfg, "temperature", "TEMPERATURE", 0.1),
                "num_predict": _int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1),
            },
        },
        timeout=_float_setting(cfg, "request_timeout", "REQUEST_TIMEOUT", 600.0, 1.0),
    )
    response.raise_for_status()
    payload = response.json()
    raw = payload.get("response", "")
    if not raw:
        raise RuntimeError("Ollama returned an empty response")
    diagnostic = f"done={payload.get('done')}; done_reason={payload.get('done_reason')}"
    return _parse_model_json(raw, "ollama", diagnostic), raw


def _call_model(prompt: str, cfg: dict, schema: dict) -> tuple[dict, str]:
    provider, model = _provider_and_model(cfg)
    if provider in _OPENAI_COMPATIBLE:
        return _call_openai_compatible(prompt, cfg, model, provider)
    if provider == "gemini":
        return _call_gemini(prompt, cfg, model, schema)
    if provider == "ollama":
        return _call_ollama(prompt, cfg, model)
    raise ProviderConfigurationError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Request wrapper: retries transient errors AND malformed JSON
# ---------------------------------------------------------------------------

def _save_failed_response(output_dir: Path, label: str, attempt: int,
                          raw: str, error: str) -> None:
    failure_dir = output_dir / "corpus_failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:100]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = failure_dir / f"{safe_label}_attempt{attempt}_{stamp}.json"
    _write_json_atomic(path, {
        "label": label, "attempt": attempt, "saved_at": _iso(_utc_now()),
        "error": error[:4000], "raw_response": raw,
    })


def _request(prompt: str, cfg: dict, schema: dict, label: str,
             output_dir: Path, scheduler: _RateScheduler, validator=None) -> dict:
    provider, model = _provider_and_model(cfg)
    retries = _int_setting(cfg, "corpus_retries", "CORPUS_RETRIES", 5, 1)
    schema_text = json.dumps(schema, ensure_ascii=False)
    base_prompt = (
        prompt + "\n\nReturn exactly one JSON object conforming to this schema. "
        "Do not include Markdown fences or explanatory text:\n" + schema_text
    )
    current_prompt = base_prompt
    last_error = None

    for attempt in range(1, retries + 1):
        scheduler.wait_turn(current_prompt)
        try:
            print(f"  {label}: {provider.upper()} / {model} request {attempt}/{retries}",
                  flush=True)
            data, _raw = _call_model(current_prompt, cfg, schema)
            if validator is not None and not validator(data):
                raise ModelJSONError(
                    f"Valid JSON did not satisfy content requirements for {label}.",
                    json.dumps(data, ensure_ascii=False),
                )
            return data
        except (ProviderConfigurationError, ProviderQuotaExhaustedError,
                ProviderCooldownError):
            raise
        except ModelJSONError as exc:
            last_error = exc
            _save_failed_response(output_dir, label, attempt, exc.raw, str(exc))
            if attempt >= retries:
                raise RuntimeError(
                    f"{provider}/{model} returned invalid JSON for {label} after "
                    f"{retries} attempts. Raw responses are in "
                    f"{output_dir / 'corpus_failures'}. Last error: {exc}"
                ) from exc
            # A focused repair instruction is more useful than blindly resending.
            current_prompt = (
                base_prompt
                + "\n\nIMPORTANT JSON REPAIR RETRY: Your previous response was invalid JSON. "
                  "Regenerate the complete object from the beginning. Ensure every string "
                  "escapes internal quotation marks, every object/array is closed, all commas "
                  "are valid, and the output is not truncated. Return JSON only."
            )
            delay = min(20.0, 2.0 * attempt) + random.uniform(0.0, 1.5)
            print(
                f"  [{provider}/{model}] invalid JSON for {label}; saved raw response; "
                f"retrying with stricter JSON instruction in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
        except Exception as exc:
            last_error = exc
            if _is_billing_exhaustion(exc) or _is_daily_quota_exhaustion(exc):
                _handle_provider_error(exc, cfg, output_dir, provider, model, attempt)
            if attempt >= retries:
                _handle_provider_error(exc, cfg, output_dir, provider, model, attempt)
                raise RuntimeError(
                    f"{provider}/{model} failed after {retries} attempts for {label}: {exc}"
                ) from exc
            delay = _handle_provider_error(exc, cfg, output_dir, provider, model, attempt)
            if delay is None:
                raise RuntimeError(
                    f"Non-retryable {provider}/{model} error for {label}: {exc}"
                ) from exc
            print(
                f"  [{provider}/{model}] request failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.1f}s", flush=True
            )
            time.sleep(delay)
    raise RuntimeError(f"Request loop exited unexpectedly: {last_error}")


# ---------------------------------------------------------------------------
# Prompts and schemas
# ---------------------------------------------------------------------------

BLUEPRINT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "client": {"type": "OBJECT", "properties": {
            "pseudonym": {"type": "STRING"}, "age": {"type": "INTEGER"},
            "occupation_or_role": {"type": "STRING"},
            "cultural_and_family_context": {"type": "STRING"},
            "relevant_history": {"type": "ARRAY", "items": {"type": "STRING"}},
            "strengths_and_supports": {"type": "ARRAY", "items": {"type": "STRING"}},
        }, "required": ["pseudonym", "age", "occupation_or_role",
            "cultural_and_family_context", "relevant_history", "strengths_and_supports"]},
        "presenting_problem": {"type": "STRING"},
        "beliefs": {"type": "ARRAY", "items": {"type": "STRING"}},
        "emotions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "therapy_frame": {"type": "STRING"},
        "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["title", "client", "presenting_problem", "beliefs",
                 "emotions", "therapy_frame", "source_refs"],
}

CONVERSATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sessions": {"type": "ARRAY", "minItems": 3, "items": {"type": "OBJECT", "properties": {
            "session_number": {"type": "INTEGER"}, "goal": {"type": "STRING"},
            "turns": {"type": "ARRAY", "minItems": 8, "items": {"type": "OBJECT", "properties": {
                "turn_id": {"type": "STRING"}, "speaker": {"type": "STRING", "enum": ["client", "therapist"]},
                "text": {"type": "STRING"}, "emotional_state": {"type": "STRING"},
                "belief_state": {"type": "STRING"}}, "required": ["turn_id", "speaker", "text", "emotional_state", "belief_state"]}},
            "ending_state": {"type": "STRING"}}, "required": ["session_number", "goal", "turns", "ending_state"]}},
        "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["sessions", "source_refs"],
}

ANNOTATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "turn_annotations": {"type": "ARRAY", "minItems": 1, "items": {"type": "OBJECT", "properties": {
            "turn_id": {"type": "STRING"}, "known_facts": {"type": "ARRAY", "items": {"type": "STRING"}},
            "hypotheses": {"type": "ARRAY", "items": {"type": "OBJECT"}},
            "supporting_evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
            "contradictions": {"type": "ARRAY", "items": {"type": "STRING"}},
            "confidence": {"type": "NUMBER"}, "uncertainty": {"type": "STRING"},
            "client_beliefs": {"type": "ARRAY", "items": {"type": "STRING"}},
            "therapist_decision": {"type": "STRING"}, "therapist_rationale": {"type": "STRING"},
            "intervention_consequence": {"type": "STRING"}, "emotional_change": {"type": "STRING"},
            "safety_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
        }, "required": ["turn_id", "known_facts", "hypotheses", "supporting_evidence", "contradictions",
            "confidence", "uncertainty", "client_beliefs", "therapist_decision", "therapist_rationale",
            "intervention_consequence", "emotional_change", "safety_flags"]}},
        "session_summaries": {"type": "ARRAY", "minItems": 1, "items": {"type": "OBJECT"}},
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
        "safe, non-diagnostic framing. The client object must include pseudonym, age, "
        "occupation_or_role, cultural_and_family_context, relevant_history, and "
        "strengths_and_supports; do not leave them empty. Copy source_refs exactly "
        "from the seed.\n" + json.dumps(seed, ensure_ascii=False)
    )


def _conversation_prompt(blueprint: dict, language: str, length: str,
                         modality: str) -> str:
    return (
        f"Write a realistic fictional multi-session counselling conversation in {language}. "
        f"Modality: {modality}; length profile: {length}. Return JSON with sessions, "
        "each having session_number, goal, turns array of {turn_id, speaker, text, "
        "emotional_state, belief_state}, and ending_state. Include 3-6 sessions, "
        "at least 8 substantial dialogue turns in each session, varied turn counts, client agency, ruptures/misunderstandings, unresolved material, "
        "and intervention consequences. CBT uses collaborative formulation and careful "
        "experiments; person-centered uses empathic reflection and non-directivity. Do not "
        "claim clinical efficacy, avoid diagnosing, handle risk disclosures by prioritizing "
        "immediate human safety assessment. Keep dialogue original, not copied from source.\n"
        + json.dumps(blueprint, ensure_ascii=False)
    )


def _session_annotation_prompt(session: dict, blueprint: dict,
                               language: str, session_number: int) -> str:
    return (
        f"Annotate ONLY session {session_number} of this fictional counselling case in "
        f"{language}. Return a JSON object with turn_annotations and session_summaries. "
        "For every turn in this session, annotate: turn_id, known_facts, hypotheses "
        "(at least two competing interpretations where plausible), supporting_evidence "
        "(turn IDs), contradictions, confidence (0..1), uncertainty, client_beliefs, "
        "therapist_decision, therapist_rationale, intervention_consequence, emotional_change, "
        "and safety_flags. Distinguish stated, inferred, uncertain, and contradicted claims. "
        "Do not invent turn IDs. Include exactly one session summary with session_number, "
        "goal_progress, key_belief_changes, unresolved_issues, alliance_observations, "
        "risk_and_safety_notes. Keep the output concise enough to fit in one JSON response. "
        "Do not treat hypotheses as facts.\n"
        + json.dumps({
            "blueprint": blueprint,
            "session": session,
            "session_number": session_number,
        }, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Run selection / manifest
# ---------------------------------------------------------------------------

def _make_run_directory(output_dir: Path, cfg: dict) -> Path:
    mode = str(_setting(cfg, "corpus_run_mode", "CORPUS_RUN_MODE", "resume", True)).lower()
    if mode not in ("resume", "new"):
        raise ValueError("CORPUS_RUN_MODE must be 'resume' or 'new'.")
    if mode == "resume":
        return output_dir
    run_id = str(_setting(cfg, "corpus_run_id", "CORPUS_RUN_ID", "")).strip()
    if not run_id:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("CORPUS_RUN_ID may contain only letters, numbers, dot, underscore, hyphen.")
    run_dir = output_dir / "corpus_runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"New run directory already exists and is not empty: {run_dir}. "
            "Choose a different CORPUS_RUN_ID or set CORPUS_RUN_MODE=resume."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_manifest(run_dir: Path, provider: str, model: str,
                   target_count: int, languages: list[str]) -> None:
    path = run_dir / "corpus_run_manifest.json"
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    created = old.get("created_at", _iso(_utc_now()))
    _write_json_atomic(path, {
        "created_at": created,
        "updated_at": _iso(_utc_now()),
        "provider": provider,
        "model": model,
        "target_count": target_count,
        "languages": languages,
        "record_id_pattern": "case_{1-based-index:08d}_{en|zh}",
        "count_semantics": "total target record count, not additional records",
        "completed_stages": {
            "blueprints": len(_read_jsonl(run_dir / "scenario_blueprints.jsonl")),
            "conversations": len(_read_jsonl(run_dir / "conversations.jsonl")),
            "annotation_parts": len(_read_jsonl(run_dir / "annotation_parts.jsonl")),
            "annotations": len(_read_jsonl(run_dir / "conversation_annotations.jsonl")),
        },
    })


def _validate_provider_configuration(cfg: dict, provider: str) -> None:
    if provider == "gemini" and not _setting(cfg, "gemini_api_key", "GEMINI_API_KEY", ""):
        raise ProviderConfigurationError("Set GEMINI_API_KEY in .env.")
    if provider in _OPENAI_COMPATIBLE:
        _compatible_credentials(cfg, provider)


# ---------------------------------------------------------------------------
# Main corpus generation
# ---------------------------------------------------------------------------

def generate_corpus(output_dir: Path, cfg: dict, count: int,
                    batch_size: int, languages=None):
    """Generate/resume records. count is the total target, not an increment."""
    _load_dotenv()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if count < 0:
        raise ValueError("count must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    provider, model = _provider_and_model(cfg)
    _validate_provider_configuration(cfg, provider)
    run_dir = _make_run_directory(output_dir, cfg)
    _check_persisted_cooldown(run_dir, provider, model)

    languages = list(languages or ["English", "Simplified Chinese"])
    if not languages:
        raise ValueError("At least one generation language is required.")

    seeds = _read_json(output_dir / "scenario_seeds.json")
    if not seeds:
        raise RuntimeError(
            f"scenario_seeds.json is empty or missing in {output_dir}. "
            "Run ingest, extract, and build first."
        )

    paths = {
        key: run_dir / f"{key}.jsonl"
        for key in (
            "scenario_blueprints", "conversations",
            "annotation_parts", "conversation_annotations",
        )
    }
    records = {key: _read_jsonl(path) for key, path in paths.items()}
    # Earlier generator versions checkpointed schema-shaped but empty objects.
    # Do not treat those placeholders as completed work; regenerate them and
    # append a better checkpoint (last row for each ID wins).
    records["scenario_blueprints"] = [r for r in records["scenario_blueprints"] if _valid_blueprint(r)]
    records["conversations"] = [r for r in records["conversations"] if _valid_conversation(r)]
    valid_ids = {r["record_id"] for r in records["conversations"]}
    records["annotation_parts"] = [r for r in records["annotation_parts"]
                                   if r.get("record_id") in valid_ids and r.get("turn_annotations")]
    records["conversation_annotations"] = [r for r in records["conversation_annotations"]
                                           if r.get("record_id") in valid_ids and r.get("turn_annotations")]
    indexes = {key: _index_rows(rows) for key, rows in records.items()}

    # count means the final desired corpus size. Existing record IDs are retained.
    queue = []
    for i in range(count):
        seed = seeds[i % len(seeds)]
        language = languages[i % len(languages)]
        suffix = "en" if language.lower().startswith("english") else "zh"
        queue.append((f"case_{i + 1:08d}_{suffix}", seed, language, i))

    _save_manifest(run_dir, provider, model, count, languages)
    scheduler = _RateScheduler(cfg, provider, model)
    print(
        f"[corpus] provider={provider} model={model} | run_dir={run_dir} | "
        f"target={count} | batch_size={batch_size}",
        flush=True,
    )

    for start in range(0, len(queue), batch_size):
        batch = queue[start:start + batch_size]
        print(
            f"[corpus] batch {start // batch_size + 1}, records "
            f"{start + 1}-{start + len(batch)} of {count}", flush=True
        )

        for record_id, seed, language, i in batch:
            refs = seed.get("source_refs", [])
            blueprint_row = indexes["scenario_blueprints"].get(record_id)

            if not blueprint_row:
                data = _request(
                    _blueprint_prompt(seed, language), cfg, BLUEPRINT_SCHEMA,
                    record_id + " blueprint", run_dir, scheduler,
                )
                if not _valid_blueprint({"blueprint": data, "source_refs": refs}):
                    raise RuntimeError(f"{record_id}: model returned an incomplete blueprint; it was not checkpointed")
                blueprint_row = {
                    "record_id": record_id,
                    "language": language,
                    "modality": (seed.get("modality_relevance") or ["cross_modality"])[0],
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
                modality = blueprint_row.get("modality", "cross_modality")
                lengths = ["short", "medium", "long"]
                data = _request(
                    _conversation_prompt(
                        blueprint_row["blueprint"], language,
                        lengths[i % len(lengths)], modality,
                    ),
                    cfg, CONVERSATION_SCHEMA, record_id + " dialogue",
                    run_dir, scheduler,
                )
                sessions = data.get("sessions", [])
                if not _valid_conversation({"sessions": sessions}):
                    # Retain the blueprint and let resume retry dialogue. Never
                    # checkpoint schema-shaped empty session objects.
                    raise RuntimeError(f"{record_id}: incomplete dialogue response; blueprint checkpoint preserved. Rerun to retry dialogue.")
                conversation_row = {
                    "record_id": record_id,
                    "language": language,
                    "modality": modality,
                    "sessions": sessions,
                    "source_refs": refs,
                    "is_synthetic": True,
                    "generation_provider": provider,
                    "generation_model": model,
                }
                _append(paths["conversations"], conversation_row)
                indexes["conversations"][record_id] = conversation_row

            # Annotation checkpoint granularity is one session. A quota interruption
            # during session N does not require re-annotating sessions 1..N-1.
            existing_parts = indexes["annotation_parts"]
            sessions = conversation_row.get("sessions", [])
            for session_index, session in enumerate(sessions, 1):
                part_id = f"{record_id}::session_{session_index:03d}"
                if part_id in existing_parts:
                    continue
                part_data = _request(
                    _session_annotation_prompt(
                        session, blueprint_row["blueprint"], language, session_index
                    ),
                    cfg, ANNOTATION_SCHEMA,
                    f"{record_id} session {session_index} annotations",
                    run_dir, scheduler,
                    validator=lambda result: (
                        len(result.get("turn_annotations", [])) == len(session.get("turns", []))
                        and all(a.get("turn_id") for a in result.get("turn_annotations", []))
                        and bool(result.get("session_summaries"))
                    ),
                )
                part_row = {
                    "part_id": part_id,
                    "record_id": record_id,
                    "session_number": session_index,
                    "language": language,
                    "turn_annotations": part_data.get("turn_annotations", []),
                    "session_summaries": part_data.get("session_summaries", []),
                    "source_refs": refs,
                    "generation_provider": provider,
                    "generation_model": model,
                }
                _append(paths["annotation_parts"], part_row)
                existing_parts[part_id] = part_row

            # Merge only after every session annotation has been checkpointed.
            annotation_row = indexes["conversation_annotations"].get(record_id)
            if not annotation_row:
                parts = [
                    existing_parts[f"{record_id}::session_{n:03d}"]
                    for n in range(1, len(sessions) + 1)
                    if f"{record_id}::session_{n:03d}" in existing_parts
                ]
                if len(parts) != len(sessions):
                    raise RuntimeError(
                        f"{record_id}: only {len(parts)}/{len(sessions)} annotation "
                        "sessions are checkpointed; rerun to continue."
                    )
                annotation_row = {
                    "record_id": record_id,
                    "language": language,
                    "turn_annotations": [
                        item for part in parts for item in part.get("turn_annotations", [])
                    ],
                    "session_summaries": [
                        item for part in parts for item in part.get("session_summaries", [])
                    ],
                    "source_refs": refs,
                    "is_synthetic": True,
                    "generation_provider": provider,
                    "generation_model": model,
                    "annotation_granularity": "session",
                }
                _append(paths["conversation_annotations"], annotation_row)
                indexes["conversation_annotations"][record_id] = annotation_row

            print(
                f"[corpus] {record_id} complete ({language}) via {provider}/{model}",
                flush=True,
            )

        _save_manifest(run_dir, provider, model, count, languages)

    _save_manifest(run_dir, provider, model, count, languages)
    print(
        "[corpus] generation finished. "
        f"blueprints={len(indexes['scenario_blueprints'])}, "
        f"conversations={len(indexes['conversations'])}, "
        f"annotation_parts={len(indexes['annotation_parts'])}, "
        f"annotations={len(indexes['conversation_annotations'])}",
        flush=True,
    )
    print(f"[corpus] artifacts saved in: {run_dir}", flush=True)
    return run_dir


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------

def validate_corpus(output_dir: Path):
    """Validate the corpus files in the supplied run directory."""
    output_dir = Path(output_dir)
    blueprints = _read_jsonl(output_dir / "scenario_blueprints.jsonl")
    conversations = _read_jsonl(output_dir / "conversations.jsonl")
    annotations = _read_jsonl(output_dir / "conversation_annotations.jsonl")
    errors = []

    maps = [
        _index_rows(blueprints),
        _index_rows(conversations),
        _index_rows(annotations),
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
            for session_index, session in enumerate(sessions, 1):
                if not session.get("turns"):
                    errors.append(f"{record_id}: session {session_index} without turns")
        if not annotation:
            errors.append(f"{record_id}: missing annotations")
        elif not annotation.get("turn_annotations"):
            errors.append(f"{record_id}: no turn annotations")

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
    print(f"Corpus validation: {len(errors)} error(s); {len(maps[1])} conversations")
    return not errors

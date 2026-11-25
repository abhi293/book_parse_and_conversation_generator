"""Checkpointed bilingual synthetic counselling corpus generation.

Providers: OpenAI, Gemini, Ollama, OpenRouter, NVIDIA NIM, Hugging Face,
Groq, Together, Fireworks, DeepSeek, Mistral, Cerebras, xAI, and arbitrary
OpenAI-compatible endpoints.

Provider selection: LLM_PROVIDER + provider-specific MODEL/API_KEY/BASE_URL
settings in .env. Generic custom endpoint: LLM_PROVIDER=custom,
LLM_BASE_URL=..., LLM_API_KEY=..., LLM_MODEL=...

Generation is sequential. JSONL artifacts are appended and fsynced after each
stage, so a rerun skips completed stages. RPM/TPM scheduling and persisted
cooldowns are maintained per selected provider/model.
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


# OpenAI-compatible provider defaults. These are API roots, not model names.
# Models are deliberately configured by the user because availability varies.
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
    aliases = {"hf": "huggingface", "hugging_face": "huggingface", "nvidia_nim": "nvidia"}
    provider = aliases.get(provider, provider)
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


def _provider_display(cfg: dict) -> str:
    provider, model = _provider_and_model(cfg)
    return f"{provider.upper()} ({model})"


# ---------------------------------------------------------------------------
# JSON and JSONL helpers
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
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        value = None
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
    for pattern, multiplier in (
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s)\b", 1),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|m)\b", 60),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", 3600),
        (r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(days?|d)\b", 86400),
    ):
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
        "slow_down", "service unavailable", "bad gateway",
    ))


def _cooldown_path(output_dir: Path) -> Path:
    return output_dir / "corpus_cooldown.json"


def _save_cooldown(output_dir: Path, provider: str, model: str,
                   retry_at: datetime, reason: str) -> None:
    path = _cooldown_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provider": provider, "model": model, "retry_at": _iso(retry_at),
               "reason": reason[:1500], "saved_at": _iso(_utc_now())}
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        print(f"[corpus] Existing cooldown belongs to {data.get('provider')} ({data.get('model')}); current target is {provider} ({model}).")
        return
    retry_at = _parse_datetime(data.get("retry_at", ""))
    if retry_at is None or retry_at <= _utc_now():
        _clear_cooldown(output_dir)
        return
    remaining = int((retry_at - _utc_now()).total_seconds())
    raise ProviderCooldownError(
        f"{provider}/{model} is in cooldown until {_iso(retry_at)} "
        f"(about {remaining // 3600}h {(remaining % 3600) // 60}m remaining). "
        f"Reason: {data.get('reason', 'provider quota/rate limit')}. Checkpoints are preserved."
    )


def _handle_provider_error(exc: Exception, cfg: dict, output_dir: Path,
                           provider: str, model: str, attempt: int) -> float | None:
    if _is_billing_exhaustion(exc):
        raise ProviderQuotaExhaustedError(
            f"{provider}/{model} billing/credit quota exhausted. Check provider billing and limits. Original: {exc}"
        ) from exc
    daily_hours = _float_setting(cfg, "quota_daily_cooldown_hours", "QUOTA_DAILY_COOLDOWN_HOURS", 12.0, 0.1)
    max_inline_wait = _float_setting(cfg, "max_inline_retry_wait", "MAX_INLINE_RETRY_WAIT", 900.0, 1.0)
    retry_after = _retry_after_seconds(exc)
    if _is_daily_quota_exhaustion(exc):
        delay = retry_after if retry_after is not None else daily_hours * 3600
        retry_at = _utc_now() + timedelta(seconds=delay)
        _save_cooldown(output_dir, provider, model, retry_at, str(exc))
        raise ProviderCooldownError(f"Daily quota appears exhausted for {provider}/{model}. Cooldown saved until {_iso(retry_at)}; rerun after cooldown.") from exc
    if not _is_transient_error(exc):
        return None
    base = _float_setting(cfg, "quota_backoff_base", "QUOTA_BACKOFF_BASE", 15.0, 0.1)
    maximum = _float_setting(cfg, "quota_backoff_max", "QUOTA_BACKOFF_MAX", 300.0, 1.0)
    delay = max(retry_after or 0.0, min(maximum, base * (2 ** max(0, attempt - 1))))
    delay += random.uniform(0.0, min(10.0, delay * 0.20))
    if delay > max_inline_wait:
        retry_at = _utc_now() + timedelta(seconds=delay)
        _save_cooldown(output_dir, provider, model, retry_at, str(exc))
        raise ProviderCooldownError(f"{provider}/{model} requested a long retry delay ({delay / 60:.1f} minutes). Cooldown saved until {_iso(retry_at)}. Rerun later; checkpoints are preserved.") from exc
    return delay


# ---------------------------------------------------------------------------
# Sequential RPM/TPM scheduler
# ---------------------------------------------------------------------------

class _RateScheduler:
    def __init__(self, cfg: dict, provider: str, model: str):
        prefix = provider.upper()
        self.rpm = _int_setting(cfg, "provider_rpm_limit", f"{prefix}_RPM_LIMIT", 0)
        self.tpm = _int_setting(cfg, "provider_tpm_limit", f"{prefix}_TPM_LIMIT", 0)
        self.min_interval = _float_setting(cfg, "provider_min_request_interval", f"{prefix}_MIN_REQUEST_INTERVAL", 0.0)
        if provider == "gemini" and self.min_interval <= 0:
            self.min_interval = _float_setting(cfg, "gemini_min_request_interval", "GEMINI_MIN_REQUEST_INTERVAL", 0.0)
        self.max_output = _int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1)
        self.events: deque[tuple[float, int]] = deque()
        self.last_request = 0.0
        self.provider, self.model = provider, model

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 2) // 3)

    def wait_turn(self, prompt: str) -> None:
        now = time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
        reservation = max(self._estimate_tokens(prompt), self.max_output)
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
            print(f"[corpus] scheduler: waiting {wait_seconds:.1f}s (RPM={self.rpm or 'unset'}, TPM={self.tpm or 'unset'}, reserved≈{reservation})", flush=True)
            time.sleep(wait_seconds + 0.05)
        now = time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
        self.events.append((now, reservation))
        self.last_request = now
        if self.tpm > 0 and reservation > self.tpm:
            print(f"[corpus] WARNING: one request reserves ≈{reservation} tokens, above configured TPM={self.tpm}; reduce MAX_OUTPUT_TOKENS or split the task.", flush=True)


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
    # Local/self-hosted compatible servers may intentionally use a dummy key.
    if not api_key and provider not in ("custom",):
        raise ProviderConfigurationError(f"Set {spec['key']} in .env for provider {provider}.")
    return api_key or "not-needed", str(base_url).rstrip("/") if base_url else None


def _call_openai_compatible(prompt: str, cfg: dict, model: str, provider: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK missing. Install with: pip install openai") from exc
    api_key, base_url = _compatible_credentials(cfg, provider)
    timeout = _float_setting(cfg, "request_timeout", "REQUEST_TIMEOUT", 600.0, 1.0)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    max_output = _int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1)
    temperature = _float_setting(cfg, "temperature", "TEMPERATURE", 0.1)
    request_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return exactly one valid JSON object. Do not use Markdown fences or explanatory text. Follow the schema in the user message."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_output,
    }
    # JSON mode is enabled by default. Turn it off for endpoints/models that do
    # not implement response_format=json_object: LLM_JSON_MODE=false.
    json_mode = str(_setting(cfg, "llm_json_mode", "LLM_JSON_MODE", "true")).lower() in ("1", "true", "yes", "on")
    if json_mode:
        request_args["response_format"] = {"type": "json_object"}
    # Some reasoning models reject temperature. Configure this per provider/model
    # rather than guessing from model names.
    use_temperature = str(_setting(cfg, "llm_send_temperature", "LLM_SEND_TEMPERATURE", "true")).lower() in ("1", "true", "yes", "on")
    if use_temperature:
        request_args["temperature"] = temperature
    effort = _setting(cfg, "llm_reasoning_effort", "LLM_REASONING_EFFORT", "")
    if effort:
        request_args["reasoning_effort"] = str(effort)
    response = client.chat.completions.create(**request_args)
    content = response.choices[0].message.content
    if isinstance(content, list):
        # Some compatible endpoints return content blocks.
        content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
    if not content:
        raise RuntimeError(f"{provider} returned an empty response")
    return _parse_model_json(content, provider)


def _call_gemini(prompt: str, cfg: dict, model: str, schema: dict) -> dict:
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
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")
    return _parse_model_json(response.text, "gemini")


def _call_ollama(prompt: str, cfg: dict, model: str) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests missing. Install with: pip install requests") from exc
    url = str(_setting(cfg, "ollama_url", "OLLAMA_URL", "http://localhost:11434")).strip().rstrip("/")
    response = requests.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "format": "json",
              "options": {"temperature": _float_setting(cfg, "temperature", "TEMPERATURE", 0.1),
                          "num_predict": _int_setting(cfg, "max_output_tokens", "MAX_OUTPUT_TOKENS", 8192, 1)}},
        timeout=_float_setting(cfg, "request_timeout", "REQUEST_TIMEOUT", 600.0, 1.0),
    )
    response.raise_for_status()
    raw = response.json().get("response", "")
    if not raw:
        raise RuntimeError("Ollama returned an empty response")
    return _parse_model_json(raw, "ollama")


def _parse_model_json(raw: str, provider: str) -> dict:
    """Parse JSON, tolerating common Markdown fences without accepting junk."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Recover a single top-level object if a model added leading/trailing text.
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"{provider} response did not contain a JSON object")
        result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise RuntimeError(f"{provider} returned JSON type {type(result).__name__}; expected object")
    return result


def _call_model(prompt: str, cfg: dict, schema: dict) -> dict:
    provider, model = _provider_and_model(cfg)
    if provider in _OPENAI_COMPATIBLE:
        return _call_openai_compatible(prompt, cfg, model, provider)
    if provider == "gemini":
        return _call_gemini(prompt, cfg, model, schema)
    if provider == "ollama":
        return _call_ollama(prompt, cfg, model)
    raise ProviderConfigurationError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Request wrapper
# ---------------------------------------------------------------------------

def _request(prompt: str, cfg: dict, schema: dict, label: str,
             output_dir: Path, scheduler: _RateScheduler) -> dict:
    provider, model = _provider_and_model(cfg)
    retries = _int_setting(cfg, "corpus_retries", "CORPUS_RETRIES", 8, 1)
    full_prompt = prompt + "\n\nReturn one JSON object conforming to this schema. Do not include Markdown fences or explanatory text:\n" + json.dumps(schema, ensure_ascii=False)
    for attempt in range(1, retries + 1):
        scheduler.wait_turn(full_prompt)
        try:
            print(f"  {label}: {provider.upper()} / {model} request {attempt}/{retries}", flush=True)
            return _call_model(full_prompt, cfg, schema)
        except (ProviderConfigurationError, ProviderQuotaExhaustedError, ProviderCooldownError):
            raise
        except Exception as exc:
            if _is_billing_exhaustion(exc) or _is_daily_quota_exhaustion(exc):
                _handle_provider_error(exc, cfg, output_dir, provider, model, attempt)
            if attempt >= retries:
                _handle_provider_error(exc, cfg, output_dir, provider, model, attempt)
                raise RuntimeError(f"{provider}/{model} failed after {retries} attempts for {label}: {exc}") from exc
            delay = _handle_provider_error(exc, cfg, output_dir, provider, model, attempt)
            if delay is None:
                raise RuntimeError(f"Non-retryable {provider}/{model} error for {label}: {exc}") from exc
            print(f"  [{provider}/{model}] request failed ({type(exc).__name__}: {exc}); retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Request loop exited unexpectedly")


# ---------------------------------------------------------------------------
# Prompts and schemas
# ---------------------------------------------------------------------------

BLUEPRINT_SCHEMA = {
    "type": "OBJECT", "properties": {
        "title": {"type": "STRING"}, "client": {"type": "OBJECT"},
        "presenting_problem": {"type": "STRING"},
        "beliefs": {"type": "ARRAY", "items": {"type": "STRING"}},
        "emotions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "therapy_frame": {"type": "STRING"},
        "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
    }, "required": ["title", "client", "presenting_problem", "beliefs", "emotions", "therapy_frame", "source_refs"],
}

CONVERSATION_SCHEMA = {
    "type": "OBJECT", "properties": {
        "sessions": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        "source_refs": {"type": "ARRAY", "items": {"type": "STRING"}},
    }, "required": ["sessions", "source_refs"],
}

ANNOTATION_SCHEMA = {
    "type": "OBJECT", "properties": {
        "turn_annotations": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        "session_summaries": {"type": "ARRAY", "items": {"type": "OBJECT"}},
    }, "required": ["turn_annotations", "session_summaries"],
}


def _blueprint_prompt(seed: dict, language: str) -> str:
    return (
        "Create one original fictional counselling case blueprint in " + language +
        ". Return JSON matching the schema. Client profile must be internally "
        "consistent and include nuanced history, strengths, context, multiple "
        "competing beliefs, emotions and interpersonal dynamics. Use source only "
        "as a high-level therapy concept anchor; do not reproduce book text or "
        "imply the fictional client appears in a book. Mark uncertainty. Maintain "
        "safe, non-diagnostic framing.\n" + json.dumps(seed, ensure_ascii=False)
    )


def _conversation_prompt(blueprint: dict, language: str, length: str, modality: str) -> str:
    return (
        f"Write a realistic fictional multi-session counselling conversation in {language}. "
        f"Modality: {modality}; length profile: {length}. Return JSON with sessions, "
        "each having session_number, goal, turns array of {turn_id, speaker(client|therapist), "
        "text, emotional_state, belief_state}, and ending_state. Include 3-6 sessions, "
        "varied turn counts, client agency, ruptures/misunderstandings, unresolved material, "
        "and intervention consequences. CBT uses collaborative formulation and careful "
        "experiments; person-centered uses empathic reflection and non-directivity. Do not "
        "claim clinical efficacy, avoid diagnosing, handle risk disclosures by prioritizing "
        "immediate human safety assessment. Keep dialogue original, not copied from source.\n" +
        json.dumps(blueprint, ensure_ascii=False)
    )


def _annotation_prompt(conversation: dict, blueprint: dict, language: str) -> str:
    return (
        f"Annotate this fictional counselling corpus in {language}. Return full inference "
        "annotations: for every turn record known facts, hypotheses (at least two competing "
        "interpretations where plausible), supporting evidence turn IDs, contradictions, "
        "confidence 0..1, uncertainty, client beliefs, therapist decision and rationale, "
        "intervention consequence, emotional change, safety flags. Distinguish stated, "
        "inferred, uncertain, contradicted. Never present hypotheses as facts. Add session summaries.\n" +
        json.dumps({"blueprint": blueprint, "conversation": conversation}, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Main corpus generation
# ---------------------------------------------------------------------------

def _validate_provider_configuration(cfg: dict, provider: str) -> None:
    if provider == "gemini" and not _setting(cfg, "gemini_api_key", "GEMINI_API_KEY", ""):
        raise ProviderConfigurationError("Set GEMINI_API_KEY in .env.")
    if provider in _OPENAI_COMPATIBLE:
        _compatible_credentials(cfg, provider)


def generate_corpus(output_dir: Path, cfg: dict, count: int,
                    batch_size: int, languages=None):
    """Generate/resume synthetic cases, preserving each completed stage."""
    _load_dotenv()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider, model = _provider_and_model(cfg)
    _validate_provider_configuration(cfg, provider)
    _check_persisted_cooldown(output_dir, provider, model)
    if count < 0:
        raise ValueError("count must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    languages = languages or ["English", "Simplified Chinese"]
    seeds = _read_json(output_dir / "scenario_seeds.json")
    if not seeds:
        raise RuntimeError("scenario_seeds.json is empty. Run ingest, extract, and build first.")

    paths = {key: output_dir / f"{key}.jsonl" for key in (
        "scenario_blueprints", "conversations", "conversation_annotations")}
    records = {key: _read_jsonl(path) for key, path in paths.items()}
    indexes = {key: {row.get("record_id"): row for row in rows} for key, rows in records.items()}
    queue = []
    for i in range(count):
        seed = seeds[i % len(seeds)]
        language = languages[i % len(languages)]
        suffix = "en" if language.lower().startswith("english") else "zh"
        queue.append((f"case_{i + 1:08d}_{suffix}", seed, language, i))

    scheduler = _RateScheduler(cfg, provider, model)
    print(f"[corpus] provider={provider} model={model} | queued={len(queue)} | batch_size={batch_size}", flush=True)
    for start in range(0, len(queue), batch_size):
        batch = queue[start:start + batch_size]
        print(f"[corpus] batch {start // batch_size + 1}, records {start + 1}-{start + len(batch)} of {count}", flush=True)
        for record_id, seed, language, i in batch:
            refs = seed.get("source_refs", [])
            blueprint_row = indexes["scenario_blueprints"].get(record_id)
            if not blueprint_row:
                data = _request(_blueprint_prompt(seed, language), cfg, BLUEPRINT_SCHEMA,
                                record_id + " blueprint", output_dir, scheduler)
                blueprint_row = {
                    "record_id": record_id, "language": language,
                    "modality": (seed.get("modality_relevance") or ["cross_modality"])[0],
                    "blueprint": data, "source_refs": refs, "is_synthetic": True,
                    "generation_provider": provider, "generation_model": model,
                }
                _append(paths["scenario_blueprints"], blueprint_row)
                indexes["scenario_blueprints"][record_id] = blueprint_row

            conversation_row = indexes["conversations"].get(record_id)
            if not conversation_row:
                modality = blueprint_row["modality"]
                lengths = ["short", "medium", "long"]
                data = _request(_conversation_prompt(blueprint_row["blueprint"], language,
                                  lengths[i % len(lengths)], modality), cfg, CONVERSATION_SCHEMA,
                                  record_id + " dialogue", output_dir, scheduler)
                conversation_row = {
                    "record_id": record_id, "language": language, "modality": modality,
                    "sessions": data.get("sessions", []), "source_refs": refs,
                    "is_synthetic": True, "generation_provider": provider,
                    "generation_model": model,
                }
                _append(paths["conversations"], conversation_row)
                indexes["conversations"][record_id] = conversation_row

            annotation_row = indexes["conversation_annotations"].get(record_id)
            if not annotation_row:
                data = _request(_annotation_prompt(conversation_row, blueprint_row["blueprint"], language),
                                cfg, ANNOTATION_SCHEMA, record_id + " annotations", output_dir, scheduler)
                annotation_row = {
                    "record_id": record_id, "language": language, **data,
                    "source_refs": refs, "is_synthetic": True,
                    "generation_provider": provider, "generation_model": model,
                }
                _append(paths["conversation_annotations"], annotation_row)
                indexes["conversation_annotations"][record_id] = annotation_row
            print(f"[corpus] {record_id} complete ({language}) via {provider}/{model}", flush=True)

    print(f"[corpus] generation finished. blueprints={len(indexes['scenario_blueprints'])}, conversations={len(indexes['conversations'])}, annotations={len(indexes['conversation_annotations'])}", flush=True)


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------

def validate_corpus(output_dir: Path):
    blueprints = _read_jsonl(Path(output_dir) / "scenario_blueprints.jsonl")
    conversations = _read_jsonl(Path(output_dir) / "conversations.jsonl")
    annotations = _read_jsonl(Path(output_dir) / "conversation_annotations.jsonl")
    errors = []
    maps = [{row.get("record_id"): row for row in rows} for rows in (blueprints, conversations, annotations)]
    for record_id, blueprint in maps[0].items():
        conversation, annotation = maps[1].get(record_id), maps[2].get(record_id)
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
    for name, rows in (("blueprints", blueprints), ("conversations", conversations), ("annotations", annotations)):
        ids = [row.get("record_id") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}: duplicate record_id values")
    for issue in errors:
        print("ERROR:", issue)
    print(f"Corpus validation: {len(errors)} error(s); {len(maps[1])} conversations")
    return not errors

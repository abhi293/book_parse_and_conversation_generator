# import os
# from pathlib import Path
# from dotenv import load_dotenv

# load_dotenv()

# def get_config():
#     return {
#         "provider": os.getenv("LLM_PROVIDER", "openai").lower(),
#         "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
#         "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
#         "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
#         "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
#         "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
#         "openrouter_model": os.getenv(
#                 "OPENROUTER_MODEL",
#                 "openrouter/free"),
#         "tesseract_cmd": os.getenv("TESSERACT_CMD", ""),
#         "ocr_languages": os.getenv("OCR_LANGUAGES", "eng+chi_sim+chi_tra"),
#         "input_dir": Path(os.getenv("INPUT_DIR", "./books")),
#         "output_dir": Path(os.getenv("OUTPUT_DIR", "./output")),
#         "chunk_chars": int(os.getenv("CHUNK_CHARS", "6000")),
#         "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "500")),
#         "ocr_mode": os.getenv("OCR_MODE", "auto").lower(),
#         "ocr_min_text_chars": int(os.getenv("OCR_MIN_TEXT_CHARS", "80")),
#         "max_chunks": int(os.getenv("MAX_CHUNKS_PER_RUN", "0")),
#     }



import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_config():
    """
    Central configuration loader.

    Supported LLM providers:
        - openai
        - openrouter
        - gemini
        - ollama

    Also handles OCR, chunking, and directory configuration.
    """

    return {

        # ==================================================
        # LLM PROVIDER
        # ==================================================

        "provider": os.getenv(
            "LLM_PROVIDER",
            "gemini"
        ).strip().lower(),

        # ==================================================
        # OPENAI
        # ==================================================

        "openai_api_key": os.getenv(
            "OPENAI_API_KEY",
            ""
        ).strip(),

        "openai_model": os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1-mini"
        ).strip(),

        # ==================================================
        # OPENROUTER
        # ==================================================

        "openrouter_api_key": os.getenv(
            "OPENROUTER_API_KEY",
            ""
        ).strip(),

        "openrouter_model": os.getenv(
            "OPENROUTER_MODEL",
            "openrouter/free"
        ).strip(),

        # ==================================================
        # GOOGLE GEMINI
        # ==================================================

        "gemini_api_key": os.getenv(
            "GEMINI_API_KEY",
            ""
        ).strip(),

        "gemini_model": os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        ).strip(),

        # ==================================================
        # OLLAMA
        # ==================================================

        "ollama_url": os.getenv(
            "OLLAMA_BASE_URL",
            os.getenv(
                "OLLAMA_URL",
                "http://localhost:11434"
            )
        ).rstrip("/"),

        "ollama_model": os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:14b"
        ).strip(),

        # ==================================================
        # GENERAL LLM SETTINGS
        # ==================================================

        "temperature": float(
            os.getenv(
                "TEMPERATURE",
                "0.1"
            )
        ),

        "max_output_tokens": int(
            os.getenv(
                "MAX_OUTPUT_TOKENS",
                "8192"
            )
        ),

        "request_timeout": int(
            os.getenv(
                "REQUEST_TIMEOUT",
                "600"
            )
        ),

        # ==================================================
        # OCR CONFIGURATION
        # ==================================================

        "tesseract_cmd": os.getenv(
            "TESSERACT_CMD",
            ""
        ).strip(),

        "ocr_languages": os.getenv(
            "OCR_LANGUAGES",
            "eng+chi_sim+chi_tra"
        ).strip(),

        "ocr_mode": os.getenv(
            "OCR_MODE",
            "auto"
        ).strip().lower(),

        "ocr_min_text_chars": int(
            os.getenv(
                "OCR_MIN_TEXT_CHARS",
                "80"
            )
        ),

        # ==================================================
        # INPUT / OUTPUT DIRECTORIES
        # ==================================================

        "input_dir": Path(
            os.getenv(
                "INPUT_DIR",
                "./books"
            )
        ),

        "output_dir": Path(
            os.getenv(
                "OUTPUT_DIR",
                "./output"
            )
        ),

        # ==================================================
        # TEXT CHUNKING
        # ==================================================

        "chunk_chars": int(
            os.getenv(
                "CHUNK_CHARS",
                "6000"
            )
        ),

        "chunk_overlap": int(
            os.getenv(
                "CHUNK_OVERLAP",
                "500"
            )
        ),

        # ==================================================
        # PROCESSING LIMITS
        # ==================================================

        "max_chunks": int(
            os.getenv(
                "MAX_CHUNKS_PER_RUN",
                "0"
            )
        ),

    }


# ======================================================
# OPTIONAL CONFIG VALIDATION
# ======================================================

def validate_config(cfg: dict):
    """
    Validate configuration before starting the pipeline.
    """

    supported_providers = {
        "openai",
        "openrouter",
        "gemini",
        "ollama"
    }

    provider = cfg["provider"]

    if provider not in supported_providers:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Choose one of: {', '.join(sorted(supported_providers))}"
        )

    provider_requirements = {
        "openai": [
            "openai_api_key",
            "openai_model"
        ],

        "openrouter": [
            "openrouter_api_key",
            "openrouter_model"
        ],

        "gemini": [
            "gemini_api_key",
            "gemini_model"
        ],

        "ollama": [
            "ollama_model"
        ]
    }

    for key in provider_requirements[provider]:

        if not cfg.get(key):
            raise ValueError(
                f"Missing required configuration: {key}"
            )

    if cfg["chunk_chars"] <= 0:
        raise ValueError(
            "CHUNK_CHARS must be greater than zero."
        )

    if cfg["chunk_overlap"] < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative."
        )

    if cfg["chunk_overlap"] >= cfg["chunk_chars"]:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_CHARS."
        )

    if cfg["temperature"] < 0 or cfg["temperature"] > 2:
        raise ValueError(
            "TEMPERATURE must be between 0 and 2."
        )

    if cfg["request_timeout"] <= 0:
        raise ValueError(
            "REQUEST_TIMEOUT must be greater than zero."
        )

    return True

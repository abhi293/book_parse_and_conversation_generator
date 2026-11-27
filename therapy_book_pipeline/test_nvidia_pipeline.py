import os
import traceback

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
base_url = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)

print("API key loaded:", bool(api_key))
print("Base URL:", base_url)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=600,
    max_retries=0
)

tests = [
    {
        "name": "Non-streaming basic",
        "args": {
            "max_tokens": 256,
        }
    },
    {
        "name": "Non-streaming JSON",
        "args": {
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
    },
    {
        "name": "Streaming JSON with thinking",
        "args": {
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            },
            "stream": True,
        }
    }
]

for test in tests:
    print(f"\nTesting: {test['name']}")

    try:
        args = {
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "messages": [
                {
                    "role": "system",
                    "content": "Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": (
                        'Return {"status": "success", '
                        '"provider": "NVIDIA"}'
                    )
                }
            ],
            **test["args"]
        }

        response = client.chat.completions.create(**args)

        if test["args"].get("stream"):
            content = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in response
                if chunk.choices
            )
        else:
            content = response.choices[0].message.content

        print("SUCCESS:", content)

    except Exception as exc:
        print("FAILED:", type(exc).__name__, str(exc))

        cause = exc.__cause__ or exc.__context__

        if cause:
            print("CAUSE:", type(cause).__name__, str(cause))

        traceback.print_exc()

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_data_foundry.llm import get_ollama_status, is_gemma_model


def main() -> None:
    status = get_ollama_status()
    if not status.available:
        print("Ollama service is not reachable.")
        print("Start it with: ollama serve")
        raise SystemExit(1)

    print("Ollama service is reachable.")
    print("Installed models:")
    for model in status.models:
        marker = "Gemma" if is_gemma_model(model) else "local debug"
        print(f"- {model} ({marker})")

    if not status.gemma_models:
        print()
        print("No Gemma model detected. For the hackathon, install a Gemma model, for example:")
        print("  ollama pull gemma4:e4b")
        print("or set ADF_OLLAMA_MODEL to point at your local Gemma 4 model.")
        raise SystemExit(2)

    print(f"Selected model: {status.selected_model}")


if __name__ == "__main__":
    main()

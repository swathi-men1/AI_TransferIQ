"""Bootstrap external-service configuration for TransferIQ."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_EXAMPLE = CONFIG_DIR / ".env.example"
ENV_PATH = CONFIG_DIR / ".env"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "raw" / "external_sources"


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    copied_env = False
    if ENV_EXAMPLE.exists() and not ENV_PATH.exists():
        shutil.copyfile(ENV_EXAMPLE, ENV_PATH)
        copied_env = True

    payload = {
        "config_env_created": copied_env,
        "env_path": str(ENV_PATH),
        "external_source_dir": str(EXTERNAL_DIR),
        "next_command": r".\venv\Scripts\python.exe scripts\collect_external_data.py --sync-all --write-manifest",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

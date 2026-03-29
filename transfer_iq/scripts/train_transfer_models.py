"""Train TransferIQ models on the provided raw dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_value_system import RAW_DATA_PATH, TransferValueTrainer


def main() -> None:
    trainer = TransferValueTrainer(sequence_length=8)
    summary = trainer.train(RAW_DATA_PATH)

    print("=" * 72)
    print("TRANSFERIQ MODEL TRAINING COMPLETE")
    print("=" * 72)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

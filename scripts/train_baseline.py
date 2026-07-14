"""Compatibility entry point for the BERT baseline.

The reproduction guide keeps the historical command
`python scripts/train_baseline.py --use_weibo21 ...` for the Chinese
baseline.  The actual implementation lives in `train_bert_baseline.py`.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_bert_baseline import main


if __name__ == "__main__":
    main()

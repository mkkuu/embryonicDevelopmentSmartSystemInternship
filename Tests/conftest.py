"""
Adds Training/ to sys.path so tests can `import evaluation...` the same
way scripts already do when run from within Training/ (see e.g.
Training/embeddings/build_cache.py's usage docstring for the same
convention), without needing an installed package or a per-file path
hack repeated across every test module.
"""

import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent.parent / "Training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

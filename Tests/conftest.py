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

# Training/ModelBuilder.py instantiates ConfigArgs() at import time, which
# parses sys.argv via argparse with no explicit args list. Under pytest,
# sys.argv still holds pytest's own CLI tokens (e.g. "Tests/", "-v"), which
# ConfigArgs's parser doesn't recognize, causing argparse to sys.exit(2)
# mid-collection. Truncate sys.argv to just the program name before any
# test module's import chain can reach ModelBuilder — pytest has already
# consumed its own CLI arguments by this point, so nothing pytest-side
# depends on sys.argv afterward. Production code (Training/) is untouched;
# this only affects the sys.argv this test process sees.
sys.argv = sys.argv[:1]

"""Shared fixtures for four_way tests.

The KB runner imports from ``kernelblaster`` which lives under
``third_party/kernelblaster/src``. Prepend it once for the whole
test session so the four_way unit tests can drive the runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

_KB_SRC = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "kernelblaster"
    / "src"
)
if str(_KB_SRC) not in sys.path:
    sys.path.insert(0, str(_KB_SRC))

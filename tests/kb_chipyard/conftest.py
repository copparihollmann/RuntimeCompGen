"""Make ``third_party/kernelblaster/src`` importable for the kb_chipyard tests.

The kernelblaster package isn't pip-installed in CompGen's venv (it's a
git submodule used via subprocess by the legacy CUDA adapter). For the
new in-process ``kb_chipyard`` sibling we just prepend its ``src/`` to
``sys.path`` once per test session — matches what ``scripts/run_RL.py``
inside the submodule does.
"""

from __future__ import annotations

import sys
from pathlib import Path

_KB_SRC = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "kernelblaster"
    / "src"
)
if str(_KB_SRC) not in sys.path:
    sys.path.insert(0, str(_KB_SRC))

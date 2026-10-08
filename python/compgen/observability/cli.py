"""Back-compat shim: the CLI now lives in ``compgen_tracker.cli``.

The ``compgen-gemini-usage`` console script in CompGen's pyproject is
now wired to :func:`compgen_tracker.cli.main`. Anything that
historically did ``from compgen.observability.cli import main`` still
resolves through this shim.
"""

from __future__ import annotations

from compgen_tracker.cli import *  # noqa: F401, F403
from compgen_tracker.cli import (  # explicit re-exports
    _aggregate_by,
    _event_group_key,
    _render_group_by,
    _render_status,
    _watch_footer,
    main,
)

__all__ = [
    "main",
]

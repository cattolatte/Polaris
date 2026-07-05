"""The console banner: bundled ASCII art, randomly colored per launch.

The Metasploit approach — art ships *in* the codebase and a random color scheme is
picked each launch — rather than any network API: the console must start instantly
and work offline, like everything else in Polaris.

Design Principles
-----------------
- Standard library only. Color is raw ANSI (256-color escape codes), no `rich` or
  `pyfiglet` until the art's needs outgrow this.
- Color is polite: suppressed when stdout is not a TTY or when the ``NO_COLOR``
  environment variable is set (https://no-color.org).
- Deterministic on demand: pass a seeded ``random.Random`` to make the choice
  reproducible (used by tests).
"""

from __future__ import annotations

import os
import random
import sys

from polaris.version import __version__

__all__ = ["COLOR_SCHEMES", "render_banner"]

_BLOCK = r"""
██████╗  ██████╗ ██╗      █████╗ ██████╗ ██╗███████╗
██╔══██╗██╔═══██╗██║     ██╔══██╗██╔══██╗██║██╔════╝
██████╔╝██║   ██║██║     ███████║██████╔╝██║███████╗
██╔═══╝ ██║   ██║██║     ██╔══██║██╔══██╗██║╚════██║
██║     ╚██████╔╝███████╗██║  ██║██║  ██║██║███████║
╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝
"""

# (name, banner color, tagline color) — 256-color ANSI foreground codes.
COLOR_SCHEMES: tuple[tuple[str, int, int], ...] = (
    ("north star", 220, 250),  # star gold + moonlight grey
    ("ice", 45, 153),  # arctic cyan + pale blue
    ("aurora", 84, 121),  # aurora green + mint
    ("nebula", 141, 183),  # violet + lavender
    ("ember", 208, 216),  # ember orange + warm sand
)

_RESET = "\x1b[0m"


def _color_enabled(*, force: bool | None = None) -> bool:
    """Decide whether to emit ANSI colors (TTY only, and never under NO_COLOR)."""
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def render_banner(
    rng: random.Random | None = None, *, color: bool | None = None
) -> str:
    """Render the banner with a randomly chosen color scheme.

    Parameters
    ----------
    rng : random.Random, optional
        Source of randomness for the scheme choice; pass a seeded instance for a
        reproducible banner. Defaults to the module-level ``random``.
    color : bool, optional
        Force colors on or off. By default, colors are used only when stdout is
        a TTY and ``NO_COLOR`` is unset.

    Returns
    -------
    str
        The banner, tagline, and version line, ready to print.
    """
    chooser = rng if rng is not None else random
    _name, art_code, tag_code = chooser.choice(COLOR_SCHEMES)

    tagline = "an educational, from-scratch NLP platform"
    version_line = f"v{__version__} · type `help` to begin"

    if _color_enabled(force=color):
        art = f"\x1b[38;5;{art_code}m{_BLOCK}{_RESET}"
        tag = f"\x1b[38;5;{tag_code}m        {tagline}\n        {version_line}{_RESET}"
    else:
        art = _BLOCK
        tag = f"        {tagline}\n        {version_line}"

    return f"{art}\n{tag}\n"

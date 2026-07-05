"""Unit tests for the console banner."""

from __future__ import annotations

import random

from polaris.console.banner import COLOR_SCHEMES, render_banner
from polaris.version import __version__

# --- content ---


def test_banner_contains_art_and_version() -> None:
    """The banner carries the block art, tagline, and current version."""
    banner = render_banner(random.Random(0), color=False)

    assert "██████╗" in banner
    assert "from-scratch NLP" in banner
    assert __version__ in banner


# --- color behavior ---


def test_color_off_emits_no_ansi_codes() -> None:
    """With color forced off, the banner is plain text."""
    banner = render_banner(random.Random(0), color=False)

    assert "\x1b[" not in banner


def test_color_on_emits_a_known_scheme() -> None:
    """With color forced on, the banner uses one of the bundled schemes."""
    banner = render_banner(random.Random(0), color=True)

    assert "\x1b[38;5;" in banner
    assert any(f"\x1b[38;5;{art}m" in banner for _, art, _tag in COLOR_SCHEMES)
    assert banner.rstrip().endswith("\x1b[0m")  # colors are always reset


def test_no_color_env_suppresses_color(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """NO_COLOR (https://no-color.org) disables ANSI output even on a TTY."""
    monkeypatch.setenv("NO_COLOR", "1")

    banner = render_banner(random.Random(0))

    assert "\x1b[" not in banner


# --- randomness ---


def test_same_seed_gives_same_banner() -> None:
    """A seeded RNG makes the scheme choice reproducible."""
    a = render_banner(random.Random(7), color=True)
    b = render_banner(random.Random(7), color=True)

    assert a == b


def test_schemes_vary_across_seeds() -> None:
    """Different seeds eventually pick different color schemes."""
    banners = {render_banner(random.Random(seed), color=True) for seed in range(20)}

    assert len(banners) > 1

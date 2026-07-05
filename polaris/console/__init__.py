"""
Interactive console subsystem for Polaris.

A `msfconsole`-style REPL over the inference layer: a bundled ASCII banner in a
randomly chosen color scheme, a `polaris >` prompt, and commands to load a saved
model bundle once and predict on raw text many times. Standard library only.
"""

from __future__ import annotations

from polaris.console.banner import render_banner
from polaris.console.repl import PolarisConsole, run

__all__ = ["PolarisConsole", "render_banner", "run"]

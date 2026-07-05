"""The interactive Polaris console: a `cmd.Cmd` REPL over the inference layer.

Load a saved model bundle once, then predict on raw text as many times as you
like — instant after the first load, unlike the one-shot `polaris predict`
command, which pays the model-load cost on every call.

Design Principles
-----------------
- Built on the standard library's :mod:`cmd` — the textbook module for exactly
  this shape (prompt loop, ``do_<command>`` methods, built-in ``help``). No
  framework, no new dependencies.
- A thin shell over :class:`~polaris.inference.predictor.Predictor`: the console
  parses lines and prints; all real work happens in the inference layer.
- Friendly failure: wrong paths or predicting before loading print a one-line
  hint, never a traceback.
"""

from __future__ import annotations

import cmd
from pathlib import Path

from polaris.errors import PolarisError
from polaris.inference import Predictor, load_bundle

__all__ = ["PolarisConsole", "run"]


class PolarisConsole(cmd.Cmd):
    """The `polaris >` interactive prompt.

    Parameters
    ----------
    intro : str, optional
        Text printed once at startup (the banner). ``None`` prints nothing.
    """

    prompt = "polaris > "

    def __init__(self, intro: str | None = None) -> None:
        super().__init__()
        self.intro = intro
        self._predictor: Predictor | None = None
        self._show_probs = False

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------

    def do_load(self, arg: str) -> None:
        """load <path> — load a saved model bundle (see `save_bundle`)."""
        path = arg.strip()
        if not path:
            print("usage: load <path-to-bundle>")
            return
        if not Path(path).is_file():
            print(f"no such file: {path}")
            return
        try:
            self._predictor = load_bundle(path)
        except (PolarisError, KeyError) as error:
            print(f"could not load bundle: {error}")
            return
        print(f"loaded: {path}")

    def do_predict(self, arg: str) -> None:
        """predict <text> — classify text with the loaded model."""
        text = arg.strip()
        if not text:
            print("usage: predict <text>")
            return
        if self._predictor is None:
            print("no model loaded — use `load <path>` first")
            return
        prediction = self._predictor.predict(text)
        print(prediction.label)
        if self._show_probs:
            for name, probability in prediction.probabilities.items():
                print(f"  {name}: {probability:.4f}")

    def do_probs(self, _arg: str) -> None:
        """probs — toggle per-class probability display on/off."""
        self._show_probs = not self._show_probs
        print(f"probabilities: {'on' if self._show_probs else 'off'}")

    def do_info(self, _arg: str) -> None:
        """info — show version and whether a model is loaded."""
        from polaris.version import __version__

        print(f"Polaris {__version__}")
        print(f"model loaded: {'yes' if self._predictor is not None else 'no'}")

    def do_exit(self, _arg: str) -> bool:
        """exit — leave the console."""
        return True

    def do_quit(self, _arg: str) -> bool:
        """quit — leave the console."""
        return True

    def do_EOF(self, _arg: str) -> bool:  # noqa: N802  (name fixed by cmd.Cmd)
        """Ctrl-D — leave the console."""
        print()
        return True

    # ------------------------------------------------------------------
    # niceties
    # ------------------------------------------------------------------

    def emptyline(self) -> bool:
        """Do nothing on an empty line (cmd's default repeats the last command)."""
        return False

    def default(self, line: str) -> None:
        """Point unknown commands at `help` instead of erroring loudly."""
        print(f"unknown command: {line.split()[0]!r} — type `help` for commands")


def run(bundle: str | None = None) -> None:
    """Start the interactive console.

    Parameters
    ----------
    bundle : str, optional
        A bundle to load immediately, as if `load <bundle>` were typed first.
    """
    from polaris.console.banner import render_banner

    console = PolarisConsole(intro=render_banner())
    if bundle is not None:
        console.do_load(bundle)
    console.cmdloop()

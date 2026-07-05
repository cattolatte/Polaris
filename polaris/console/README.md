# Console Module

The interactive Polaris console — a `msfconsole`-style REPL over the inference
layer: an ASCII banner in a randomly chosen color scheme, a `polaris >` prompt,
and commands to **load a model bundle once and predict many times** (instant after
the first load, unlike the one-shot `polaris predict`).

```
polaris console --model runs/imdb_pretrained_transformer/model.pt

polaris > predict a wonderful, moving film
pos
polaris > probs
probabilities: on
polaris > predict dull and lifeless
neg
  neg: 0.9873
  pos: 0.0127
polaris > help
```

## Public surface

- `run(bundle=None)` (`repl.py`) — start the console, optionally pre-loading a
  bundle. Wired to the `polaris console` CLI command.
- `PolarisConsole` (`repl.py`) — the `cmd.Cmd` REPL: `load`, `predict`, `probs`,
  `info`, `exit`/`quit`, plus the built-in `help`.
- `render_banner(rng=None, *, color=None)` (`banner.py`) — the block-letter banner
  in one of the bundled ANSI color schemes, chosen randomly per launch.

## Design notes

- **Standard library only** (`cmd` + raw ANSI). The banner ships in the codebase
  and is randomized locally — the Metasploit pattern, no network, no API.
- Color respects `NO_COLOR` (https://no-color.org) and non-TTY output.
- The console is a thin shell: all real work happens in `polaris.inference`.

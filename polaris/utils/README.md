# Utils Module

Small, dependency-light helpers shared across the stack: reproducibility, device
selection, and logging. Nothing here is NLP-specific; each helper exists because a
real caller needed it.

## Public surface

- `set_seed(seed)` (`seed.py`) — seed Python, NumPy, and PyTorch for reproducible
  runs.
- `resolve_device(device=None)` (`device.py`) — pick the compute device: an explicit
  string, or auto-select Apple Silicon **MPS**, then CUDA, else CPU.
- `module_device(module)` (`device.py`) — the device a model's parameters live on.
- `get_logger()` (`logging.py`) — the shared, consistently formatted `polaris`
  logger used by the training and pretraining loops.

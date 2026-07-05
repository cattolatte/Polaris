# Deployment Module

A thin **HTTP serving layer** over a trained Polaris model. It loads a saved
bundle into a [`Predictor`](../inference/) and exposes it as a FastAPI app, so a
model can be invoked as a real service — the final step of the end-to-end story.

The module is a deliberately thin adapter: all model and inference logic lives in
`polaris.inference`; this layer only maps HTTP to a `Predictor` call.

## Public surface

- `create_app(bundle_path)` (`app.py`) — build a FastAPI app serving a bundle,
  with two routes:
  - `GET /health` — liveness probe (`{"status": "ok"}`).
  - `POST /predict` — body `{"text": "..."}` → `{"label", "label_id", "probabilities"}`.
- `PredictRequest` / `PredictResponse` (`app.py`) — the Pydantic request/response
  schemas (Pydantic is already a Polaris dependency).

## Optional dependency

FastAPI and uvicorn are an optional extra, imported lazily (own-the-interface:
they never leak into the model or inference layers). Install with:

```bash
uv sync --extra serving
```

Calling `create_app` without the extra raises a `MissingDependencyError` with an
install hint, never a bare `ImportError`.

## Running

From the CLI (see also the repository `Dockerfile`):

```bash
polaris serve --model runs/imdb_pretrained_transformer/model.pt --port 8000
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "a wonderful, moving film"}'
# {"label":"pos","label_id":1,"probabilities":{"neg":0.02,"pos":0.98}}
```

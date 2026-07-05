# Serve a trained Polaris model over HTTP.
#
# Build:  docker build -t polaris-serve .
# Run:    docker run --rm -p 8000:8000 -v "$PWD/runs:/models:ro" \
#             -e POLARIS_MODEL=/models/imdb_pretrained_transformer/model.pt \
#             polaris-serve
#
# The model bundle is provided at run time (mounted read-only), so the image is
# model-agnostic. Then: curl -s localhost:8000/predict \
#   -H 'content-type: application/json' -d '{"text": "a wonderful film"}'

FROM python:3.12-slim

# uv for fast, reproducible installs (and CPU-only torch wheels on Linux, per the
# pytorch-cpu index configured in pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

# Install the package with the torch (CPU) and serving extras only.
RUN uv sync --extra torch --extra serving

ENV POLARIS_MODEL=/models/model.pt
EXPOSE 8000

CMD ["sh", "-c", "uv run polaris serve --model \"$POLARIS_MODEL\" --host 0.0.0.0 --port 8000"]

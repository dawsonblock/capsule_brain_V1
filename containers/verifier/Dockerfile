FROM python:3.11-slim
RUN python -m pip install --no-cache-dir \
    "pytest>=8.0" \
    "ruff>=0.6" \
    "mypy>=1.11"
WORKDIR /workspace
USER 65532:65532

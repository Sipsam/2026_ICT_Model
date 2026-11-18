ARG ICT_BASE=python:3.11-slim
FROM ${ICT_BASE}

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /ict
RUN mkdir -p /ict/jobs && chmod 1777 /ict/jobs
COPY pyproject.toml README.md /tmp/ict-build/
COPY src/ict_pipeline /tmp/ict-build/src/ict_pipeline
RUN python -m pip install --no-cache-dir '/tmp/ict-build[preprocess]'

ENTRYPOINT ["ict"]
CMD ["--help"]

# syntax=docker/dockerfile:1
# The Tez server: `tez serve` in front of a llama.cpp server. Build from the repository root:
#   docker build -f docker/Dockerfile -t tez .
#   docker run --rm -p 127.0.0.1:8787:8787 -e TEZ_BACKEND=http://host.docker.internal:8080 tez
# Configuration is by environment (TEZ_BACKEND, TEZ_TEMPLATE, TEZ_SCHEMAS, TEZ_PORT, TEZ_API_KEY, ... and *_FILE
# secrets): see docker/entrypoint.py and docs/DOCKER.md. compose.yaml runs it next to llama.cpp.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY tez ./tez
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:${PYTHON_VERSION}-slim
# Optional extras, e.g. --build-arg EXTRAS=fit to run `tez fit` / `tez suggest` in the container (adds scikit-learn).
ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TEZ_HOST=0.0.0.0 \
    TEZ_PORT=8787 \
    TEZ_DATA_DIR=/data
COPY --from=build /wheels /wheels
RUN set -eu; wheel="$(ls /wheels/*.whl)"; pip install --no-cache-dir "${wheel}${EXTRAS:+[$EXTRAS]}" \
    && rm -rf /wheels \
    && groupadd --system --gid 10001 tez \
    && useradd --system --uid 10001 --gid tez --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin tez \
    && mkdir -p /schemas /data \
    && chown tez:tez /data
COPY docker/entrypoint.py /usr/local/lib/tez/entrypoint.py
USER 10001:10001
WORKDIR /data
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:%s/healthz' % os.environ.get('TEZ_PORT', '8787'), timeout=4)"]
ENTRYPOINT ["python", "/usr/local/lib/tez/entrypoint.py"]

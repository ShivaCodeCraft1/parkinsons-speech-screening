# Inference + training image. The trained model is NOT baked in: mount it (see docker-compose.yml) or COPY it in your own build.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt
COPY api api
COPY src src
COPY configs configs
COPY frontend frontend
RUN useradd -m appuser && mkdir -p models/current && chown -R appuser /app
USER appuser
ENV PD_MODEL_DIR=/app/models/current PD_ENABLE_DEMO_MODE=0
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

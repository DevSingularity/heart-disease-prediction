# Heart Disease Risk Prediction - ML application + trainer image
# Base image is pinned; Python 3.11.10 on Debian Bookworm (slim).
FROM python:3.11.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- dependency layer (cached while requirements.txt is unchanged) ---------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- application code layer ------------------------------------------------
COPY . .

# Non-root user + mount points for the named volumes. Docker copies these
# directories (with this ownership) into empty named volumes on first use,
# so the appuser-owned processes can write dataset/model/report files.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data /app/artifacts /app/reports \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# Default: run the Streamlit app. The `trainer` service overrides this with
# `python -m src.models.train` in compose.yaml.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

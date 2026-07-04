FROM python:3.12-slim

WORKDIR /srv

# Layer-cache the install: metadata + source first, artifacts/app after.
COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY artifacts ./artifacts
COPY app ./app

EXPOSE 8000 8501

# Default command serves the API; the app service overrides it in compose.
CMD ["uvicorn", "neighbourhood_pulse.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from app code — Docker caches this
# layer, so changing your code later won't force a slow dependency reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and the trained model
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
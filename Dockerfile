# Stage 1: Build Stage (for installing dependencies)
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application Stage (copying actual code)
COPY . .

# Expose the port that the application will listen on.
# Use an environment variable for flexibility.
# NOTE: Cloud Run injects the PORT env var automatically.
EXPOSE 8080

# Command to run your FastAPI application.
# Use the $PORT environment variable, which Cloud Run provides.

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

# Dockerfile

# Stage 1: Builder — installs dependencies
# Using CUDA 12.8 base to match your training environment
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS builder

# Prevents interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1

WORKDIR /app

# Copy and install requirements first (Docker layer caching)
# If requirements don't change, this layer is cached on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — lean final image
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Environment variables with defaults
# Override these when running the container
ENV ADAPTER_REPO="Sid9797/querycraft-phi3-sql"
ENV MOCK_MODEL="false"
ENV PORT=8000

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
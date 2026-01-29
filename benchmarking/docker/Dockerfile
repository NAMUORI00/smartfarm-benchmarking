# ERA-SmartFarm-RAG Benchmarking Docker Image
# 
# Build:
#   docker build -t era-benchmark:latest -f docker/Dockerfile ..
#
# Run:
#   docker run --rm -v $(pwd)/output:/app/output era-benchmark:latest
#   docker run --rm --gpus all -e DEVICE=cuda -v $(pwd)/output:/app/output era-benchmark:latest

FROM python:3.10-slim

LABEL maintainer="ERA-SmartFarm Team"
LABEL description="Benchmarking environment for ERA-SmartFarm-RAG"
LABEL version="1.0"

# Build arguments
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG CACHE_DIR=/root/.cache

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEVICE=cpu \
    HF_HOME=${CACHE_DIR}/huggingface \
    TRANSFORMERS_CACHE=${CACHE_DIR}/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=${CACHE_DIR}/sentence_transformers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: Skip llama-cpp-python for benchmarking (not needed for retrieval experiments)
RUN grep -v "llama-cpp-python" requirements.txt > requirements_benchmark.txt && \
    pip install --no-cache-dir -r requirements_benchmark.txt && \
    pip install --no-cache-dir pyyaml tqdm && \
    rm requirements_benchmark.txt

# Pre-download embedding model to bake into image
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBEDDING_MODEL}')" && \
    echo "Embedding model cached: ${EMBEDDING_MODEL}"

# Copy project files
COPY benchmarking/ ./benchmarking/

# Create output directory
RUN mkdir -p /app/output

# Default config location
ENV BENCHMARK_CONFIG=/app/benchmarking/config/benchmark_config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import benchmarking; print('OK')" || exit 1

# Entry point
ENTRYPOINT ["python", "-m", "benchmarking.run_benchmark"]

# Default arguments (can be overridden)
CMD ["--config", "/app/benchmarking/config/benchmark_config.yaml"]

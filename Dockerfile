# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Playwright
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user with proper home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Install Playwright browsers as root (this ensures proper installation)
# RUN playwright install chromium
# RUN playwright install-deps chromium

# Create necessary directories and set permissions
RUN mkdir -p /home/appuser/.cache /home/appuser/.local \
    && mkdir -p /app/schemas /app/data \
    && chown -R appuser:appuser /home/appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /usr/local/lib/python3.11/site-packages/playwright

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Switch to non-root user
USER appuser

RUN playwright install

# Set environment variable for Crawl4AI cache directory
ENV CRAWL4AI_CACHE_DIR=/home/appuser/.cache/crawl4ai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "src.main"]
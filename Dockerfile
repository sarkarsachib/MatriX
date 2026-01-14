# Multi-stage Dockerfile for Sathik AI Direction Mode

# Stage 1: Python backend
FROM python:3.11-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_direction_mode.txt .
RUN pip install --no-cache-dir -r requirements_direction_mode.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs

# Expose FastAPI port
EXPOSE 8000

# Stage 2: Node.js frontend
FROM node:18-alpine as frontend

WORKDIR /app/web_ui

# Copy package files
COPY web_ui/package*.json ./
COPY web_ui/package-lock.json* ./

# Install all dependencies (including devDependencies for build)
RUN npm ci

# Copy source code
COPY web_ui/ .

# Build the application
RUN npm run build

# Stage 3: Final production image
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY --from=backend /app/requirements_direction_mode.txt .
RUN pip install --no-cache-dir -r requirements_direction_mode.txt

# Copy application code (excluding node_modules)
COPY --from=backend /app /app

# Copy built frontend
COPY --from=frontend /app/web_ui/dist /app/web_ui/dist

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV FRONTEND_PATH=/app/web_ui/dist

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
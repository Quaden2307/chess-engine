# Multi-stage build for Chess Bot

# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY chess-frontend/package*.json ./
RUN npm install
COPY chess-frontend/ ./
RUN npm run build

# Stage 2: Python Backend with Frontend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend_improved.py .
COPY engine.py .
COPY api/ ./api/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/build ./chess-frontend/build

# Expose port
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=backend_improved.py
ENV PYTHONUNBUFFERED=1
ENV PORT=5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/api/health')" || exit 1

# Run the application
CMD ["python", "backend_improved.py"]

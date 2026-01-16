FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (gcc needed for some python libs)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Combine pip installs to reduce layers
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ibapi

# Copy source code
COPY src/ ./src/
COPY exported-assets/config.py ./exported-assets/
COPY watchlist.csv .
COPY checkpoints/ ./checkpoints/

# Setup Environment
ENV PYTHONPATH=/app/src
ENV IB_HOST=ib-gateway
ENV IB_PORT=4001

# Command to run
CMD ["python", "src/trading_system.py"]

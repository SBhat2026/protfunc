# Use Python 3.11 as requested
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user/app

# Copy requirements and install
# Note: your requirements.txt already includes --extra-index-url for CPU torch
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files (static/, .pth, .pkl, .csv, server.py)
COPY --chown=user . .

# Start the server using uvicorn
# Hugging Face looks for port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]


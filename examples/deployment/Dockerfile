# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Ensure markov-agent is installed. 
# If not on PyPI, you might need to copy the source:
# COPY markov_agent /app/markov_agent
# ENV PYTHONPATH=/app

# Copy application code
COPY . .

# Environment variables
ENV PORT=8080

# Run the web server
CMD ["python", "main.py"]

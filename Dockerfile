FROM python:3.10-slim

WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default env vars — overridden by hackathon infra at runtime
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""

# Port for HF Spaces health check ping
EXPOSE 7860

# Run inference script (required by hackathon)
CMD ["python", "inference.py"] 
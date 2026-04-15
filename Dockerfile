FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user that Hugging Face Spaces requires
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install them securely
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose port (Hugging Face Spaces uses port 7860 by default)
EXPOSE 7860

# Run the FastAPI server natively (bypassing run.py strictly for Docker networking)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]

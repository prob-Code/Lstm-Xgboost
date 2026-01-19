# Use a Python base image
FROM python:3.10-slim

# Create a non-root user (HF best practice)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user . /app

# Expose the default Space port
EXPOSE 7860

# Start your Flask server
CMD ["python", "hybrid_api.py"]

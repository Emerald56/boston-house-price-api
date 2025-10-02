# Use Python 3.9 base image
FROM python:3.13-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file 
COPY requirements.txt .

# Install system Dependencies
RUN  pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on 
EXPOSE 8000

# Run Gunicorn server to serve as FastAPI application
CMD [ "gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
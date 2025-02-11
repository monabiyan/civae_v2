# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install system dependencies (including zip)
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libz-dev \
    pkg-config \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Ensure entrypoint.sh is executable
RUN chmod +x /app/entrypoint.sh

# (Optional) Declare the output folder as a volume so it’s preserved if mounted
VOLUME ["/app/output"]

# Set the entrypoint script as the container command
CMD ["/app/entrypoint.sh"]

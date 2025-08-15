FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    sudo \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install Python packages
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy requirements and install
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY requirements-dev.txt requirements-dev.txt
ARG ENV
RUN if [ "$ENV" = "dev" ]; then pip install -r requirements-dev.txt; fi

# Set working directory and assign ownership
WORKDIR /root/workspace

# Expose SSH and Jupyter ports
EXPOSE 8888

# Configure prefect server url
RUN if [ "$ENV" = "dev" ]; then prefect config set PREFECT_API_URL=${PREFECT_API_URL}; fi

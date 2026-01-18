FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/x86_64-linux-gnu/libgdal.so




# Add symlink for GEOS compatibility (libgeos_c.so)
RUN ln -sf /usr/lib/x86_64-linux-gnu/libgeos_c.so /usr/lib/x86_64-linux-gnu/libgeos.so

# Set GDAL and GEOS environment variables for Django GIS (override for both locations)
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    GDAL_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgdal.so \
    GEOS_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgeos_c.so

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA 13.0-compatible PyTorch and JAX wheels. These are
# installed explicitly to select the appropriate CUDA builds and
# avoid hash mismatches from pre-pinned wheels in requirements.
# NOTE: Host must have NVIDIA drivers and the NVIDIA Container Toolkit.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
        torch==2.1.0 || true && \
    pip install --no-cache-dir jax==0.4.23 || true && \
    pip install --no-cache-dir --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib==0.4.23+cuda13.cudnn86 || true

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p staticfiles media

# Default command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

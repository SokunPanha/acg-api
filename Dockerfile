# Backend: FastAPI + ffmpeg (pydub & video rendering need the ffmpeg binary)
# CUDA runtime base so ffmpeg can use NVENC (h264_nvenc) for GPU video encoding.
# The container toolkit mounts the host's NVIDIA driver libs at runtime; this image
# provides the CUDA userspace that NVENC needs. (CPU still works via libx264 fallback.)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ffmpeg is required by pydub (audio) and the video renderer.
# Ubuntu's ffmpeg is built with NVENC enabled and loads libnvidia-encode at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first for better layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# no --reload in the container; mount code + add --reload in compose for dev if wanted
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

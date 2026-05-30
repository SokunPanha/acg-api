# Backend: FastAPI + ffmpeg (pydub & video rendering need the ffmpeg binary)
FROM python:3.12-slim

# ffmpeg is required by pydub (audio) and the video renderer
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# no --reload in the container; mount code + add --reload in compose for dev if wanted
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

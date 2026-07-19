# Official image ships torch/torchaudio already built and matched against the
# bundled CUDA 12.1 runtime, so there's no separate pip install of torch needed
# (that path was hanging/misresolving CUDA sub-package versions on RunPod's builders).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Models are lazily downloaded by the handler on first invocation (see
# get_tts_model/get_translate_model) rather than baked in at build time --
# baking them in kept hitting transient HF Hub download failures on RunPod's
# build workers. Trade-off: first cold start is slower, but builds are
# reliable and fast.
COPY serverless_handler.py .

CMD ["python", "serverless_handler.py"]

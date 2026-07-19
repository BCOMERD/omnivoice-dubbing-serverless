# Official image ships torch/torchaudio already built and matched against the
# bundled CUDA 12.1 runtime, so there's no separate pip install of torch needed
# (that path was hanging/misresolving CUDA sub-package versions on RunPod's builders).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both models at build time so cold starts don't re-download
# them every time a new worker spins up. Done before COPY serverless_handler.py
# so editing the handler doesn't bust this expensive cache layer.
# Split into separate unbuffered steps so a failure pinpoints which download broke.
RUN python -u -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice'); print('OMNIVOICE_DOWNLOAD_OK')"
RUN python -u -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); print('NLLB_MODEL_DOWNLOAD_OK')"
RUN python -u -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', src_lang='arb_Arab'); print('NLLB_TOKENIZER_DOWNLOAD_OK')"

COPY serverless_handler.py .

CMD ["python", "serverless_handler.py"]

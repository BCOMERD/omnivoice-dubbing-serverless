FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both models at build time so cold starts don't re-download
# them every time a new worker spins up. Done before COPY serverless_handler.py
# so editing the handler doesn't bust this expensive cache layer.
RUN python -c "from omnivoice import OmniVoice; OmniVoice.from_pretrained('k2-fsa/OmniVoice')"
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); \
    AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M', src_lang='arb_Arab')"

COPY serverless_handler.py .

CMD ["python", "serverless_handler.py"]

"""
Vietnamese ASR API — Whisper fine-tuned (LoRA merged) on VLSP2020-100h.
FastAPI service: nhận file audio, trả về transcript tiếng Việt + latency.
"""
import io
import time
import torch
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ─── Load model MỘT LẦN khi khởi động server ────────────────────────────
MODEL_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

print(f"[startup] Loading model on {DEVICE} ...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE).eval()
forced_ids = processor.get_decoder_prompt_ids(language="vietnamese", task="transcribe")
print("[startup] Model ready.")

app = FastAPI(
    title="Vietnamese ASR API",
    description="Whisper-small fine-tuned (LoRA) for Vietnamese speech-to-text.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Health check — để load balancer / người dùng kiểm tra server sống."""
    return {"status": "ok", "device": DEVICE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Nhận 1 file audio, trả transcript tiếng Việt kèm latency."""
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ. Chấp nhận: {', '.join(ALLOWED_EXT)}",
        )

    try:
        audio_bytes = await file.read()
        # librosa tự decode mp3/wav/... và resample về 16kHz (bắt buộc cho Whisper)
        audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được file audio: {e}")

    if len(audio_array) == 0:
        raise HTTPException(status_code=400, detail="File audio rỗng.")

    start = time.time()
    inputs = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(inputs, forced_decoder_ids=forced_ids, max_new_tokens=225)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    latency = time.time() - start

    return {
        "transcription": text,
        "latency_seconds": round(latency, 3),
        "audio_duration_seconds": round(len(audio_array) / 16000, 2),
        "device": DEVICE,
    }
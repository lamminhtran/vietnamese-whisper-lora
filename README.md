# Vietnamese Whisper (LoRA)

Nhận dạng tiếng nói tiếng Việt bằng `openai/whisper-small` fine-tune với LoRA trên VLSP2020-100h.
Repo này là phần **serving**: FastAPI + Docker cho model đã merge adapter.

| | WER | CER |
|---|:---:|:---:|
| whisper-small (baseline) | 33.55% | 21.92% |
| **+ LoRA (model này)** | **15.81%** | **8.20%** |

Test trên 2.822 mẫu. Chỉ huấn luyện 0.36% tham số (884.736 / 244M).

**Liên kết:** [🤗 Model](https://huggingface.co/lamminhtran/whisper-vi-lora) · [📓 Notebook (Kaggle)](https://www.kaggle.com/code/lmminhtrn/train-speedai-22th9) · [🐙 Code](https://github.com/lamminhtran/vietnamese-whisper-lora)

---

## Quick start

Trọng số model (~922MB) không nằm trong repo — tải từ Hugging Face trước:

```bash
pip install huggingface_hub
hf download lamminhtran/whisper-vi-lora --local-dir ./model
```

**Docker (khuyến nghị):**

```bash
docker build -t whisper-vi-api .
docker run -d --name whisper-vi --restart unless-stopped -p 8000:8000 whisper-vi-api
```

**Local:**

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

> Yêu cầu: Python 3.10–3.12, `torch>=2.6` (bắt buộc cho `transformers` 5.x), `ffmpeg` (Docker đã có sẵn).

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/transcribe` | Upload audio → transcript |
| `GET` | `/docs` | Swagger UI để test |

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@audio.mp3"
```

```json
{
  "transcription": "...",
  "latency_seconds": 3.24,
  "audio_duration_seconds": 2.0,
  "device": "cpu"
}
```

Audio hỗ trợ: `.wav .mp3 .flac .m4a .ogg` (tự resample về 16kHz mono).

## Dữ liệu

- **VLSP2020-100h** (VinBigData) — ~100 giờ, 56.427 mẫu. Mirror Parquet: [`doof-ferb/vlsp2020_vinai_100h`](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h). License CC-BY-4.0.
- Chia train/val/test 90/5/5 (seed 42) → 50.784 / 2.821 / 2.822 mẫu.
- Audio chuẩn hóa 16kHz mono.

## Huấn luyện

LoRA đóng băng trọng số gốc, chỉ train các ma trận rank thấp chèn vào lớp attention — giữ được
kiến thức đa ngôn ngữ của Whisper, checkpoint adapter chỉ vài MB.

| | |
|---|---|
| Base | `openai/whisper-small` (244M) |
| LoRA | r=8, α=16, dropout=0.05, target `q_proj` / `v_proj` |
| Optimizer | lr 5e-4, warmup 100 steps, 2 epochs |
| Batch | 16 hiệu quả (8 × grad accum 2) |
| Khác | fp16, gradient checkpointing, SpecAugment, early stopping theo val WER (patience 3) |
| Hardware | Kaggle 2× Tesla T4 (16GB) |

Metric: WER và CER. CER quan trọng với tiếng Việt do thanh điệu và dấu phụ. Val WER (~15%) khớp
test WER (15.81%) → không overfit.

## Phân tích lỗi

So với baseline, model fine-tune cải thiện ở 3 nhóm:

- **Chuẩn hóa số → chữ:** `"2019"` → `"hai nghìn không trăm mười chín"`; `"940 phút"` → `"chín giờ bốn mươi phút"`
- **Thanh điệu & âm vị** (giảm CER): `"cô hồi"` → `"cơ hội"`; `"toàn phủ hủy"` → `"thành phố huế"`
- **Từ vay mượn:** `"festival"` → `"phét ti van"` (theo phong cách phiên âm của dataset)

Câu dài nhiều số và tên riêng: WER 47% → 5%.

**Hạn chế:** audio khó (phương ngữ, ca dao) và từ hiếm / tên riêng vẫn sai nhiều. Whisper giới hạn
30s mỗi lần — audio dài hơn cần cắt đoạn.

## Hướng phát triển

- Shallow fusion với LM (KenLM / n-gram) khi decode.
- Beam search thay greedy decoding.
- Fine-tune từ PhoWhisper thay vì Whisper gốc.
- ONNX / quantization / GPU serving (hiện chạy CPU, ~3–15s/file 10s).
- Ablation LoRA rank (8 / 16 / 32) và target modules.

## Cấu trúc

```
whisper-vi-api/
├── app.py            # FastAPI: load model + /health, /transcribe
├── requirements.txt
├── Dockerfile        # production: non-root user, HEALTHCHECK
├── .dockerignore
├── .gitignore
└── model/            # trọng số — tải từ Hugging Face, không commit
    └── README.md
```

Phần data prep + train + đánh giá nằm trong [notebook Kaggle](https://www.kaggle.com/code/lmminhtrn/train-speedai-22th9).

## Stack

Python · PyTorch · Transformers · PEFT · FastAPI · Docker · librosa

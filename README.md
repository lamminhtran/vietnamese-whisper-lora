# Vietnamese Speech Recognition — Fine-tuning Whisper with LoRA

Fine-tune mô hình **Whisper-small** của OpenAI cho tác vụ nhận dạng tiếng nói tiếng Việt (ASR), sử dụng kỹ thuật **LoRA** (Low-Rank Adaptation) trên bộ dữ liệu **VLSP2020-100h** của VinBigData. Dự án đi trọn chuỗi từ xử lý dữ liệu → huấn luyện → đánh giá → triển khai thành API phục vụ production.

> **Kết quả chính:** WER giảm từ **33.55% → 15.81%** (giảm 53%), CER giảm từ **21.92% → 8.20%** (giảm 63%) trên tập test 2.822 mẫu — chỉ huấn luyện 0.36% số tham số nhờ LoRA.

> 📓 **Notebook huấn luyện (data prep + train + đánh giá):** [Kaggle — train-speedai-22th9](https://www.kaggle.com/code/lmminhtrn/train-speedai-22th9)
> 🤗 **Model đã merge (trọng số):** [huggingface.co/lamminhtran/whisper-vi-lora](https://huggingface.co/lamminhtran/whisper-vi-lora)
> Toàn bộ quá trình fine-tune được chạy trên Kaggle (2× Tesla T4). Repo này chứa phần **triển khai (serving)** của model đã merge.

---

## 1. Tổng quan

| Hạng mục | Chi tiết |
|---|---|
| Bài toán | Automatic Speech Recognition (ASR) tiếng Việt |
| Mô hình nền | `openai/whisper-small` (244M tham số) |
| Kỹ thuật fine-tune | LoRA (r=8, α=16) trên các lớp `q_proj`, `v_proj` |
| Augmentation | SpecAugment |
| Dữ liệu | VLSP2020-100h (VinBigData) — 56.427 mẫu, ~100 giờ |
| Phần cứng huấn luyện | Kaggle, 2× Tesla T4 (16GB) |
| Triển khai | FastAPI + Docker |

## 2. Dữ liệu

- **Nguồn:** VLSP2020-100h ASR challenge dataset của VinBigData (VinGroup), công bố công khai cho cộng đồng tại hội thảo VLSP 2020. Lấy qua bản mirror Parquet trên Hugging Face (`doof-ferb/vlsp2020_vinai_100h`), gốc từ VinBigData. License: CC-BY-4.0.
- **Tiền xử lý:** chuẩn hóa audio về 16kHz mono; tự chia train/validation/test theo tỉ lệ 90/5/5 với seed cố định (42) để đảm bảo tái lập và tránh data leakage.
- **Thống kê sau chia:** Train 50.784 — Validation 2.821 — Test 2.822 mẫu.

## 3. Phương pháp

**Vì sao LoRA:** thay vì cập nhật toàn bộ 244M tham số (tốn VRAM, dễ catastrophic forgetting), LoRA đóng băng trọng số gốc và chỉ huấn luyện các ma trận rank thấp chèn vào lớp attention — chỉ **884.736 tham số (0.36%)**. Checkpoint nhẹ ~vài MB, giữ được kiến thức đa ngôn ngữ gốc của Whisper.

**Cấu hình huấn luyện chính:**
- LoRA: r=8, alpha=16, dropout=0.05, target=[`q_proj`, `v_proj`]
- Learning rate 5e-4, warmup 100 steps, 2 epochs
- Batch hiệu quả 16 (batch 8 × gradient accumulation 2)
- fp16 mixed precision, gradient checkpointing (tối ưu VRAM)
- SpecAugment (che ngẫu nhiên dải tần/thời gian trên spectrogram)
- Early stopping theo validation WER (patience=3)

**Đánh giá:** WER (Word Error Rate) và CER (Character Error Rate) — CER đặc biệt quan trọng với tiếng Việt do hệ thống 6 thanh điệu và dấu phụ.

## 4. Kết quả

| Mô hình | WER | CER |
|---|---|---|
| Whisper-small (baseline, chưa fine-tune) | 33.55% | 21.92% |
| **Whisper-small + LoRA (fine-tuned)** | **15.81%** | **8.20%** |
| **Cải thiện tuyệt đối** | **−17.74** | **−13.72** |

Kết quả validation (~15% WER) khớp với kết quả test (15.81%), cho thấy mô hình **không overfit** — huấn luyện ổn định.

## 5. Phân tích lỗi (Error Analysis)

So sánh output baseline và model fine-tune cho thấy 3 nhóm cải thiện rõ rệt:

**(a) Chuẩn hóa số thành chữ** — model học được quy ước viết số bằng chữ của domain:
- `"2019"` → `"hai nghìn không trăm mười chín"` ✓
- `"940 phút"` → `"chín giờ bốn mươi phút"` ✓

**(b) Sửa dấu thanh điệu & âm vị** (nguyên nhân chính giảm CER):
- `"ờ sao lại thế"` → `"ơ sao lại thế"` ✓
- `"cô hồi"` → `"cơ hội"` ✓
- `"toàn phủ hủy"` → `"thành phố huế"` ✓

**(c) Từ vay mượn / phiên âm:**
- `"festival"` → `"phét ti van"` ✓ (đúng phong cách phiên âm của dataset)

**Ví dụ tiêu biểu** (câu dài nhiều số và tên riêng): WER baseline 47% → fine-tuned **5%**.

**Hạn chế (Limitations):** với audio khó (phương ngữ, ca dao như "thằng bờm") hoặc từ hiếm/tên riêng, cả hai model vẫn sai nhiều — model fine-tune có cải thiện nhưng chưa giải quyết triệt để.

## 6. Triển khai (Deployment)

Mô hình được merge adapter LoRA vào trọng số gốc (`merge_and_unload`) thành một model Whisper hoàn chỉnh, rồi phục vụ qua **FastAPI** và đóng gói **Docker**.

### 6.1. Tải model

Trọng số model (~922MB) **không** được commit lên GitHub. Tải về thư mục `model/` trước khi chạy:

```bash
pip install huggingface_hub
hf download lamminhtran/whisper-vi-lora --local-dir ./model
```

Thư mục `model/` sau khi tải cần có đủ: `config.json`, `generation_config.json`,
`model.safetensors`, `preprocessor_config.json`, `processor_config.json`,
`tokenizer.json`, `tokenizer_config.json`. Chi tiết xem [model/README.md](model/README.md).

### 6.2. Chạy bằng Docker (khuyến nghị)

```bash
docker build -t whisper-vi-api .
docker run -d --name whisper-vi --restart unless-stopped -p 8000:8000 whisper-vi-api

# Giao diện test tự sinh: http://localhost:8000/docs
# Hoặc gọi trực tiếp:
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.mp3"
```

### 6.3. Chạy local (không Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

API trả về JSON gồm `transcription`, `latency_seconds`, `audio_duration_seconds`, `device`.
Endpoint `GET /health` dùng cho health check. Định dạng audio hỗ trợ: `.wav .mp3 .flac .m4a .ogg`.

> **Lưu ý môi trường:** cần Python 3.10–3.12, `torch>=2.6` (bắt buộc cho `transformers` 5.x),
> và `ffmpeg` để decode mp3/m4a (Docker image đã cài sẵn).

## 7. Hướng phát triển (Future work)

- Kết hợp **Language Model (KenLM/n-gram)** khi decode để sửa lỗi ngữ pháp/từ vựng.
- **Beam search** thay greedy decoding để giảm WER.
- Fine-tune tiếp từ **PhoWhisper** (model Vietnamese ASR SOTA) thay vì Whisper gốc.
- Tối ưu inference cho production: **ONNX / quantization / GPU serving** (hiện chạy CPU, ~3–15s/file 10s).
- Ablation study về LoRA rank (r=8 vs 16 vs 32) và target modules.

## 8. Cấu trúc thư mục

```
whisper-vi-api/
├── app.py              # FastAPI server (load model + endpoints /health, /transcribe)
├── requirements.txt    # dependencies (torch cài riêng — xem mục 6.3)
├── Dockerfile          # đóng gói production (non-root user, HEALTHCHECK)
├── .dockerignore
├── .gitignore
├── model/              # trọng số model — tải từ Hugging Face, KHÔNG commit
│   └── README.md       # hướng dẫn tải model
└── README.md
```

> Phần **data prep + huấn luyện + đánh giá** nằm trong notebook Kaggle (xem link ở đầu),
> không nằm trong repo serving này.

## 9. Công nghệ sử dụng

Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · Weights & Biases · FastAPI · Docker · librosa

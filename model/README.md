# model/

Thư mục này chứa trọng số mô hình Whisper tiếng Việt. Các file sau **phải**
được đặt vào đây trước khi chạy API (chúng là file nhị phân/mô hình thật, không
thể tạo tự động):

- `config.json`
- `generation_config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `processor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`

## Cách tải model

Ví dụ tải một model Whisper tiếng Việt từ Hugging Face:

```bash
pip install huggingface_hub

# Thay <repo-id> bằng model bạn muốn, ví dụ:
#   openai/whisper-small  (đa ngôn ngữ, hỗ trợ tiếng Việt)
#   vinai/PhoWhisper-small (fine-tune cho tiếng Việt)
huggingface-cli download <repo-id> \
    --local-dir ./model \
    --local-dir-use-symlinks False
```

Hoặc bằng Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="<repo-id>", local_dir="model")
```

Sau khi các file trên có mặt, chạy:

```bash
pip install -r requirements.txt
python app.py
# hoặc build Docker: docker build -t whisper-vi-api . && docker run -p 8000:8000 whisper-vi-api
```

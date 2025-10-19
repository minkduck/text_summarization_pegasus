import argparse
import os
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline

def detect_latest_checkpoint(parent_dir: str) -> str:
    """
    Tự động chọn checkpoint có global_step lớn nhất trong parent_dir.
    Ví dụ: FINAL/checkpoint-3000, FINAL/checkpoint-3682 -> trả về checkpoint-3682
    """
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Parent dir không tồn tại: {parent_dir}")

    cand = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                cand.append((step, full))
            except Exception:
                pass
    if not cand:
        raise ValueError(f"Không tìm thấy thư mục checkpoint-* trong {parent_dir}")
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]

def load_model_tokenizer(model_dir: str):
    """
    Load tokenizer + model từ thư mục checkpoint đã fine-tune.
    Nếu thiếu file tokenizer trong checkpoint, có thể đổi sang base tokenizer:
      base_tok = 'google/pegasus-samsum'  hoặc  'google/pegasus-large'
    """
    print(f"🔹 Loading from: {model_dir}")
    tokenizer = PegasusTokenizer.from_pretrained(model_dir)
    model = PegasusForConditionalGeneration.from_pretrained(model_dir)
    return model, tokenizer

def build_pipeline(model, tokenizer, device_preference: str):
    # Chọn thiết bị
    if device_preference == "cpu":
        device = -1
    else:
        device = 0 if torch.cuda.is_available() else -1
    print("🔹 Device:", "GPU" if device == 0 else "CPU")

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return summarizer

def summarize_texts(summarizer, texts, max_len, min_len, num_beams, do_sample):
    outputs = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        out = summarizer(
            t,
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            do_sample=do_sample
        )[0]["summary_text"]
        outputs.append((t, out))
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Run Pegasus summarization from fine-tuned checkpoint.")
    parser.add_argument("--parent_dir", type=str, default="FINAL",
                        help="Thư mục cha chứa các checkpoint-xxxx (local) hoặc chính là checkpoint folder.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Đường dẫn trực tiếp tới checkpoint. Nếu bỏ trống, script tự chọn checkpoint mới nhất trong parent_dir.")
    parser.add_argument("--text", type=str, default=None,
                        help="Một đoạn hội thoại để tóm tắt (ưu tiên hơn --file).")
    parser.add_argument("--file", type=str, default=None,
                        help="Đường dẫn file .txt (mỗi dòng là 1 input cần tóm tắt).")
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--do_sample", action="store_true", help="Bật sampling (mặc định tắt).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"], help="Chạy GPU nếu có (auto) hoặc bắt buộc CPU.")

    args = parser.parse_args()

    # Xác định đường dẫn checkpoint
    if args.model_dir is None:
        # Nếu parent_dir chính là một checkpoint hợp lệ thì dùng luôn
        if os.path.isdir(args.parent_dir) and os.path.isfile(os.path.join(args.parent_dir, "config.json")):
            model_dir = args.parent_dir
        else:
            # Ngược lại, tìm checkpoint-* mới nhất trong parent_dir
            model_dir = detect_latest_checkpoint(args.parent_dir)
    else:
        model_dir = args.model_dir

    # Load model + tokenizer
    model, tokenizer = load_model_tokenizer(model_dir)
    summarizer = build_pipeline(model, tokenizer, args.device)

    # Chuẩn bị input
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        if not os.path.isfile(args.file):
            raise ValueError(f"Không tìm thấy file: {args.file}")
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines()]
    else:
        # ví dụ mặc định
        texts = ["""A: Hey, did you book the venue for Friday?
B: Not yet, I'm comparing prices.
A: We need it by tonight.
B: Okay, I'll finalize the booking."""]

    # Chạy tóm tắt
    results = summarize_texts(
        summarizer, texts,
        max_len=args.max_len, min_len=args.min_len,
        num_beams=args.num_beams, do_sample=args.do_sample
    )

    # In kết quả
    print("\n===== RESULTS =====")
    for i, (src, hyp) in enumerate(results, 1):
        print(f"\n[# {i}] SOURCE:\n{src}\n---\nSUMMARY:\n{hyp}")

if __name__ == "__main__":
    main()

import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
from modeling_qwen3_embed import Qwen3ForEmbedding
import os

def load_json_or_jsonl(path):
    texts, text_ids = [], []
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)
        if first_line.startswith("{"):
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    text = item.get("query") or item.get("text")
                    text_id = item.get("text_id", None)
                    if text is not None:
                        texts.append(text)
                        text_ids.append(text_id)
        else:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    text = item.get("query") or item.get("text")
                    text_id = item.get("text_id", None)
                    if text is not None:
                        texts.append(text)
                        text_ids.append(text_id)
    return texts, text_ids


@torch.no_grad()
def encode(model, tokenizer, texts, max_len, device, batch_size=64):
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs, mode="embedding")
        embeddings = outputs["embeddings"]  # Last token + L2 normalized
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--encode_in_path", type=str, required=True)
    parser.add_argument("--encoded_save_path", type=str, required=True)
    parser.add_argument("--p_max_len", type=int, default=128)
    parser.add_argument("--q_max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)


    if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
        print("[INFO] Detected LoRA adapter, loading base + adapter...")
        base_model = Qwen3ForEmbedding.from_pretrained(
            args.tokenizer_name,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)


        print("[INFO] Merging LoRA weights into base model for faster inference...")
        model = model.merge_and_unload()
        # --------------------
    else:
        print("[INFO] Detected full model, loading directly...")
        model = Qwen3ForEmbedding.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True
        )

    model = model.to(device)

    texts, text_ids = load_json_or_jsonl(args.encode_in_path)

    is_query = args.encode_in_path.endswith(".json")
    max_len = args.q_max_len if is_query else args.p_max_len

    embeddings = encode(model, tokenizer, texts, max_len, device, batch_size=args.batch_size)

    torch.save((embeddings, text_ids), args.encoded_save_path)
    print(f"[\u2713] Saved {len(embeddings)} embeddings to: {args.encoded_save_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path

from transformers import AutoTokenizer


MODEL_ID = "openai-community/gpt2"
OUTPUT_DIR = Path("/home/ubuntu/datasets/tokenizer/gpt2")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(OUTPUT_DIR)

    print("vocab size:", len(tok))
    print("EOS id:", tok.eos_token_id)
    print("EOS token:", repr(tok.eos_token))


if __name__ == "__main__":
    main()

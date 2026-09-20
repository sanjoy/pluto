#!/usr/bin/env python3

from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER_DIR = Path("/home/ubuntu/datasets/tokenizer/gpt2")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    try:
        while True:
            line = input("text: ")
            encoding = tok.encode(line, add_special_tokens=False)
            decoded = tok.decode(encoding, clean_up_tokenization_spaces=False)

            print("encoding:", encoding)
            print("decoded:", repr(decoded))

            if decoded != line:
                raise RuntimeError(f"round trip failed: {line!r} != {decoded!r}")

            print("round trip: ok")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

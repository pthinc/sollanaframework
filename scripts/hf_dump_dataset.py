import os
from datasets import load_dataset

def main():
    ds = load_dataset("pthinc/turkish_english_general_dataset")
    split = ds.get("train") or next(iter(ds.values()))
    sample = split[0]
    text_field = None
    for k, v in sample.items():
        if isinstance(v, str):
            text_field = k
            break
    if text_field is None:
        raise RuntimeError("No text field found")

    out_path = os.path.join("data", "huggingface_turkish_english_sample.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subset = split.select(range(min(200, len(split))))
    with open(out_path, "w", encoding="utf-8") as f:
        for row in subset:
            f.write(row[text_field].replace("\n", " ") + "\n")
    print("saved", out_path)

if __name__ == "__main__":
    main()

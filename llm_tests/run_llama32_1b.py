"""
Quick runner for meta-llama/Llama-3.2-1B with minimal dependencies.
Supports CPU by default; will use GPU if torch sees one (CUDA/ROCm/DirectML build).
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device_map() -> str | None:
    # Let HF/accelerate choose if CUDA/ROCm/DirectML is available; else None (CPU)
    if torch.cuda.is_available():
        return "auto"
    # ROCm may surface as cuda in PyTorch ROCm builds; handled above
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "auto"
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", default="BCE pipeline için kısa bir özet yaz.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device_map = get_device_map()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    inputs = tok(args.prompt, return_tensors="pt")
    if device_map:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

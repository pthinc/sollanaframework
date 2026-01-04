"""
Run Llama-3.2-1B (or another HF causal LM) and feed its hidden-state embedding into SystemIntegrator for BCE scoring.
This is a lightweight bridge: we pool the last hidden state mean as `phi_vec` and run integrator.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import sys
from pathlib import Path

# add Behavioral Consciousness to path
BC_ROOT = Path(__file__).resolve().parents[1] / "Behavioral Consciousness"
sys.path.append(str(BC_ROOT))
from system_integrator import SystemIntegrator  # type: ignore


def get_device_map() -> str | None:
    if torch.cuda.is_available():
        return "auto"
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "auto"
    except Exception:
        pass
    return None


def pooled_embedding(model, tok, text: str, device_map: str | None):
    inputs = tok(text, return_tensors="pt")
    if device_map:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1]  # (1, seq, hidden)
    pooled = hidden.mean(dim=1).squeeze(0)  # (hidden,)
    return pooled.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", default="BCE pipeline için kısa bir özet yaz.")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()

    device_map = get_device_map()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Generate a short completion (not used for embedding)
    inputs = tok(args.prompt, return_tensors="pt")
    if device_map:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=0.7, top_p=0.9)
    completion = tok.decode(gen[0], skip_special_tokens=True)

    # Pool hidden states for embedding
    phi_vec_full = pooled_embedding(model, tok, completion, device_map)
    # fit to integrator superneuron dim (8) by truncating or padding
    target_dim = 8
    if phi_vec_full.shape[0] >= target_dim:
        phi_vec = phi_vec_full[:target_dim]
    else:
        phi_vec = np.zeros(target_dim, dtype=float)
        phi_vec[: phi_vec_full.shape[0]] = phi_vec_full

    # Build behavior/context for SystemIntegrator
    behavior = {
        "phi_vec": phi_vec.tolist(),
        "phi": 0.8,
        "history_count": 4,
        "decay_rate": 0.05,
        "meta": {"ethical_tag": "approved"},
        "decay_level": 0.1,
        "resonance": 0.7,
        "char_sal": 0.8,
        "lambda_base": 0.1,
    }
    ctx = {
        "user_id": "llm_user",
        "user_type": "bağ_kurucu",
        "context_vec": np.random.rand(len(phi_vec)),
        "char_vector": np.random.rand(len(phi_vec)),
        "recent_matrix_for_iso": np.random.rand(20, len(phi_vec)),
        "clustering_matrix": np.random.rand(30, len(phi_vec)),
        "interaction_features": {"engagement": 0.6},
        "allow_auto": True,
        "text": completion,
    }

    integ = SystemIntegrator()
    res = integ.process_behavior(behavior, ctx)
    print("=== LLM completion ===")
    print(completion)
    print("=== BCE outputs (subset) ===")
    print({
        "bce": res.get("bce"),
        "output": res.get("output"),
        "trust": res.get("trust"),
        "ethic_guard": res.get("ethic_guard"),
        "ego_balance": res.get("ego_balance"),
    })


if __name__ == "__main__":
    main()

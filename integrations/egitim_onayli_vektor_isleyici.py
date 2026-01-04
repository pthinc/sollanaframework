# vektor_isleyici.py
"""Vector processor with transformer backend; now uses backend helpers and HF token/cache."""

import numpy as np
from typing import Optional

from backends import ensure_backend, load_transformer_model, get_hf_token, get_cache_dir

class VektorIsleyici:
    def __init__(self, model_name: str = "bert-base-uncased", backend_name: Optional[str] = None,
                 hf_token: Optional[str] = None, cache_dir: Optional[str] = None):
        backend = ensure_backend(backend_name)
        if not backend.name.startswith("torch"):
            raise RuntimeError("VektorIsleyici requires a torch backend for transformers.")

        token = hf_token or get_hf_token()
        cache = get_cache_dir(cache_dir)
        self.model, self.tokenizer = load_transformer_model(model_name, token=token, cache_dir=str(cache))
        self.model.eval()
        self.backend = backend

    def hazirla_girdi(self, phi_vec: np.ndarray, embeddings: np.ndarray):
        """
        phi_vec: duygu vektörü (ör. 1xN)
        embeddings: davranış vektörü (ör. 1xM)
        returns: birleşik tensor
        """
        import torch

        phi_tensor = torch.tensor(phi_vec, dtype=torch.float32)
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        combined = torch.cat([phi_tensor, emb_tensor], dim=-1)
        return combined.unsqueeze(0)  # batch dimension

    def aktivasyonlari_al(self, input_text: str):
        """
        input_text: transformer için metin girişi
        returns: hidden states ve attention weights
        """
        import torch

        inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs.hidden_states  # list of layers
        attentions = outputs.attentions        # list of layers
        return {
            "hidden_states": hidden_states,
            "attentions": attentions
        }

    def girdi_ve_aktivasyon(self, phi_vec: np.ndarray, embeddings: np.ndarray, input_text: str):
        """
        Tam işlem: vektörleri hazırla + metinle aktivasyonları al
        """
        combined_vec = self.hazirla_girdi(phi_vec, embeddings)
        activations = self.aktivasyonlari_al(input_text)
        return {
            "combined_input": combined_vec,
            "activations": activations
        }

# Örnek kullanım
if __name__ == "__main__":
    processor = VektorIsleyici()
    phi_vec = np.random.rand(8)
    embeddings = np.random.rand(16)
    input_text = "The system is stable and emotionally resonant."

    result = processor.girdi_ve_aktivasyon(phi_vec, embeddings, input_text)
    print("Birleşik girdi şekli:", result["combined_input"].shape)
    print("Hidden state katman sayısı:", len(result["activations"]["hidden_states"]))
    print("Attention katman sayısı:", len(result["activations"]["attentions"]))

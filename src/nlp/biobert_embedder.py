"""Sentence embeddings from BioBERT (mean pooling over the last hidden layer).

BioBERT (dmis-lab/biobert-v1.1) is a pretrained *encoder*; it is not a fine-tuned NER model,
so it is used here only to produce dense text representations. Embeddings are cached on disk.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class BioBertEmbedder:
    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1", max_length: int = 128,
                 batch_size: int = 32, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state                     # [B, T, H]
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)     # [B, T, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)     # mean over real tokens
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)

    def embed_cached(self, texts: list[str], cache_dir: str | Path) -> np.ndarray:
        key = hashlib.sha1(("\n".join(texts) + self.model_name + str(self.max_length)).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"biobert_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self.embed(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

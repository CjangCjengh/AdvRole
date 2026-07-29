from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.common import load_jsonl

logger = logging.getLogger(__name__)

def scenario_text(s: dict) -> str:
    parts = [s.get("target_character", ""), s.get("target_profile", ""),
             s.get("scenario", ""), s.get("history_text", "")]
    return "\n".join(str(p) for p in parts if p)

class LocalEmbedder:

    def __init__(self, model_path: str, device: str = "cuda", max_len: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.device = device
        self.max_len = max_len

    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
                last = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().float().numpy())
        return np.concatenate(vecs, axis=0)

class APIEmbedder:
    def __init__(self, api_base: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=60.0)
        self.model = model

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(
                model=self.model, input=texts[i:i + batch_size])
            arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)
            vecs.append(arr)
        return np.concatenate(vecs, axis=0)

def intra_pool_distance(emb: np.ndarray, subsample: int = 2000,
                        seed: int = 0) -> float:
    n = len(emb)
    if n > subsample:
        idx = np.random.RandomState(seed).choice(n, subsample, replace=False)
        emb = emb[idx]
    sims = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    return float(1.0 - sims[iu].mean())

def embedding_drift(orig_emb: np.ndarray, rw_emb: np.ndarray) -> float:
    sims = np.sum(orig_emb * rw_emb, axis=1)
    return float((1.0 - sims).mean())

def new_clusters(orig_emb: np.ndarray, rw_emb: np.ndarray, k: int = 30,
                 seed: int = 0) -> str:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(orig_emb)
    combined = np.concatenate([orig_emb, rw_emb], axis=0)
    labels = km.predict(combined)
    n_orig = len(orig_emb)
    counts = {}
    for i, lab in enumerate(labels):
        c = counts.setdefault(int(lab), [0, 0])
        c[0 if i < n_orig else 1] += 1
    fresh = sum(1 for o, r in counts.values() if r > 0 and r / (o + r) >= 0.5)
    return f"{fresh}/{k}"

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, help="S0 pool jsonl")
    p.add_argument("--rewritten", required=True, help="rewrites jsonl (with original_id)")
    p.add_argument("--pool", default=None,
                   help="merged pool jsonl; IPD is measured on this "
                        "(defaults to original+rewritten)")
    p.add_argument("--embed-model", required=True, help="local path or API model id")
    p.add_argument("--embed-api-base", default="local",
                   help="'local' = mean-pooled encoder; else /v1 endpoint")
    p.add_argument("--embed-api-key", default="EMPTY")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    original = load_jsonl(args.original)
    rewritten = load_jsonl(args.rewritten)
    pool = load_jsonl(args.pool) if args.pool else original + rewritten
    logger.info(f"original={len(original)} rewritten={len(rewritten)} pool={len(pool)}")

    if args.embed_api_base == "local":
        embedder = LocalEmbedder(args.embed_model, device=args.device)
    else:
        embedder = APIEmbedder(args.embed_api_base, args.embed_api_key, args.embed_model)

    orig_by_id = {s["id"]: s for s in original}
    paired_rw = [s for s in rewritten if s.get("original_id") in orig_by_id]

    pool_emb = embedder.embed([scenario_text(s) for s in pool])
    orig_emb = embedder.embed([scenario_text(s) for s in original])
    rw_emb = embedder.embed([scenario_text(s) for s in paired_rw])
    rw_src_emb = embedder.embed(
        [scenario_text(orig_by_id[s["original_id"]]) for s in paired_rw])

    result = {
        "IPD": round(intra_pool_distance(pool_emb), 3),
        "ED": round(embedding_drift(rw_src_emb, rw_emb), 3),
        "NC": new_clusters(orig_emb, rw_emb, k=args.k),
        "n_pool": len(pool), "n_original": len(original),
        "n_rewritten_paired": len(paired_rw),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {args.output}")

if __name__ == "__main__":
    main()

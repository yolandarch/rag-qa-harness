"""
Embedding lewat API yang kompatibel dengan format OpenAI (endpoint /embeddings).

Dipakai oleh ingest.py (embed dokumen) dan rag.py (embed pertanyaan).
Nama model "chroma-default" berarti: pakai embedding bawaan Chroma (baseline).

Konfigurasi di .env (opsional, kalau kosong memakai LLM_BASE_URL / LLM_API_KEY):
    EMB_BASE_URL=...
    EMB_API_KEY=...
"""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING = "chroma-default"


def slug(model: str) -> str:
    """Nama model -> potongan aman untuk nama koleksi Chroma."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def collection_name(domain: str, embedding: str, chunk_size: int, overlap: int) -> str:
    # Baseline tetap memakai nama lama supaya report sebelumnya masih bisa direproduksi.
    if embedding == DEFAULT_EMBEDDING:
        return f"{domain}_cs{chunk_size}_ov{overlap}"
    return f"{domain}_{slug(embedding)}_cs{chunk_size}_ov{overlap}"


def _client() -> OpenAI:
    load_dotenv(ROOT / ".env", override=True)
    base = os.getenv("EMB_BASE_URL") or os.environ["LLM_BASE_URL"]
    key = os.getenv("EMB_API_KEY") or os.environ["LLM_API_KEY"]
    return OpenAI(base_url=base, api_key=key, timeout=60, max_retries=2)


def embed(texts: list[str], model: str, batch_size: int = 50, sleep: float = 35.0,
          max_retries: int = 5) -> list[list[float]]:
    """Embed banyak teks per batch.
    Free tier Gemini menghitung SETIAP teks sebagai satu request (limit 100/menit),
    jadi 50 teks per batch + jeda 35 detik ~ 86 teks/menit, di bawah limit."""
    from openai import RateLimitError

    client = _client()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except RateLimitError:
                if attempt == max_retries:
                    raise
                wait = 30 * attempt
                print(f"  [429] kena rate limit, tunggu {wait}s lalu coba lagi ({attempt}/{max_retries})")
                time.sleep(wait)

        data = resp.data
        # Sebagian provider (mis. Gemini) tidak mengisi `index`; kalau begitu andalkan urutan respons.
        if all(getattr(d, "index", None) is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        if len(data) != len(batch):
            raise RuntimeError(f"Batch berisi {len(batch)} teks tapi API mengembalikan {len(data)} embedding")
        vectors.extend(d.embedding for d in data)

        done = min(start + batch_size, len(texts))
        print(f"  embedded {done}/{len(texts)}")
        if done < len(texts):
            time.sleep(sleep)
    return vectors
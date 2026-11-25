"""Ingest PDF ke Chroma dengan chunk per N karakter (chunking lama; yang dipakai sekarang ingest_pasal.py).

    python -m app.ingest --domain banking --embedding gemini-embedding-001 [--chunk-size N --overlap N] [--reset]
"""

import argparse
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

from app.embeddings import DEFAULT_EMBEDDING, collection_name, embed

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / ".chroma"


def clean_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap harus lebih kecil dari chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # mundur ke akhir kalimat atau baris terdekat
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + chunk_size // 2:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_pdf_chunks(pdf_path: Path, chunk_size: int, overlap: int):
    # chunk tidak melintasi halaman, jadi nomor halamannya pasti benar
    reader = PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue  # halaman kosong atau hasil scan
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            # ID tetap, jadi ingest ulang tidak bikin duplikat dan bisa dirujuk dari golden set
            chunk_id = f"{pdf_path.stem}-p{page_num}-c{i}"
            yield chunk_id, chunk, {"source": pdf_path.name, "page": page_num, "chunk": i}


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF ke ChromaDB")
    parser.add_argument("--domain", default="banking", help="nama folder di data/, mis. banking")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                        help='nama model embedding di API, atau "chroma-default" untuk baseline')
    parser.add_argument("--reset", action="store_true", help="hapus koleksi lama sebelum ingest")
    args = parser.parse_args()

    docs_dir = ROOT / "data" / args.domain / "docs"
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Tidak ada PDF di {docs_dir}. Download dokumennya dulu (lihat data/{args.domain}/SOURCES.md).")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(args.domain, args.embedding, args.chunk_size, args.overlap)
    if args.reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name,
        metadata={"embedding_model": args.embedding, "chunk_size": args.chunk_size, "overlap": args.overlap},
    )

    total = 0
    for pdf in pdfs:
        ids, texts, metas = [], [], []
        for chunk_id, text, meta in load_pdf_chunks(pdf, args.chunk_size, args.overlap):
            ids.append(chunk_id)
            texts.append(text)
            metas.append(meta)
        if not ids:
            print(f"[WARN] {pdf.name}: tidak ada teks yang bisa diekstrak (PDF hasil scan?)")
            continue

        vectors = None if args.embedding == DEFAULT_EMBEDDING else embed(texts, args.embedding)

        for b in range(0, len(ids), 100):
            kwargs = {"ids": ids[b:b + 100], "documents": texts[b:b + 100], "metadatas": metas[b:b + 100]}
            if vectors is not None:
                kwargs["embeddings"] = vectors[b:b + 100]
            collection.upsert(**kwargs)
        print(f"[OK] {pdf.name}: {len(ids)} chunk")
        total += len(ids)

    print(f"\nSelesai. {total} chunk di koleksi '{name}' (embedding: {args.embedding})")


if __name__ == "__main__":
    main()
"""
Ingest PDF dokumen ke ChromaDB.

Pemakaian (dari root repo):
    python -m app.ingest --domain banking
    python -m app.ingest --domain banking --chunk-size 800 --overlap 150 --reset

Alur: PDF -> teks per halaman -> chunk (dengan overlap) -> simpan ke Chroma
beserta metadata (nama file, nomor halaman), supaya jawaban bisa dilacak sumbernya.
"""

import argparse
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / ".chroma"


def clean_text(text: str) -> str:
    """Rapikan teks hasil ekstraksi PDF: gabungkan spasi berlebih, buang baris kosong beruntun."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Potong teks per karakter dengan overlap. Usahakan memotong di batas kalimat/baris."""
    if overlap >= chunk_size:
        raise ValueError("overlap harus lebih kecil dari chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # mundur ke titik/baris baru terdekat agar kalimat tidak terpotong di tengah
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
    """Hasilkan (id, teks, metadata) untuk setiap chunk. Chunk tidak melintasi halaman,
    supaya metadata nomor halaman selalu akurat."""
    reader = PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue  # halaman kosong / hasil scan tanpa teks
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            # ID deterministik: ingest ulang tidak bikin duplikat, dan bisa dirujuk di golden set
            chunk_id = f"{pdf_path.stem}-p{page_num}-c{i}"
            yield chunk_id, chunk, {"source": pdf_path.name, "page": page_num, "chunk": i}


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF ke ChromaDB")
    parser.add_argument("--domain", default="banking", help="nama folder di data/, mis. banking")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--reset", action="store_true", help="hapus koleksi lama sebelum ingest")
    args = parser.parse_args()

    docs_dir = ROOT / "data" / args.domain / "docs"
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Tidak ada PDF di {docs_dir}. Download dokumennya dulu (lihat data/{args.domain}/SOURCES.md).")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = f"{args.domain}_cs{args.chunk_size}_ov{args.overlap}"
    if args.reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # koleksi belum ada
    # Catatan: embedding default Chroma adalah model kecil berbahasa Inggris.
    # Sengaja dipakai sebagai baseline; nanti dibandingkan dengan model multilingual.
    collection = client.get_or_create_collection(collection_name)

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
        # upsert per batch agar aman untuk dokumen besar
        for b in range(0, len(ids), 100):
            collection.upsert(ids=ids[b:b + 100], documents=texts[b:b + 100], metadatas=metas[b:b + 100])
        print(f"[OK] {pdf.name}: {len(ids)} chunk")
        total += len(ids)

    print(f"\nSelesai. {total} chunk di koleksi '{collection_name}' ({CHROMA_DIR})")


if __name__ == "__main__":
    main()
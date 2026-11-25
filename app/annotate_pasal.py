"""Tambah metadata `pasal` dan `bagian` ke chunk yang sudah ada di Chroma, tanpa embed ulang.

PDF dipotong ulang dengan chunk_size/overlap dari metadata koleksi, jadi ID chunk-nya sama.

    python -m app.annotate_pasal --domain banking --collection banking_gemini-embedding-001_cs800_ov150 [--dry-run]
"""

import argparse
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

from app.ingest_pasal import CHROMA_DIR, ROOT, chunk_text, clean_text

PASAL_HEADING = re.compile(r"^\s*Pasal\s+(\d+[A-Z]?)\s*$", re.MULTILINE)
PENJELASAN_HEADING = re.compile(r"^\s*PENJELASAN\s*$", re.MULTILINE)


def pasal_label(start_pasal: str | None, inner: list[str]) -> str:
    nums = ([start_pasal] if start_pasal else []) + [p for p in inner if p != start_pasal]
    if not nums:
        return ""
    return f"Pasal {nums[0]}" if len(nums) == 1 else f"Pasal {nums[0]}-{nums[-1]}"


def annotate_pdf(pdf_path: Path, chunk_size: int, overlap: int):
    """(chunk_id, metadata) per chunk, dipotong sama persis seperti ingest.py."""
    reader = PdfReader(str(pdf_path))
    current_pasal, bagian = None, "Batang Tubuh"
    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        headings = [(m.start(), m.group(1)) for m in PASAL_HEADING.finditer(text)]
        penjelasan_pos = [m.start() for m in PENJELASAN_HEADING.finditer(text)]

        cursor = 0
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            start = text.find(chunk, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk)
            cursor = max(cursor, start + 1)

            # pasal di awal chunk = judul pasal terakhir sebelum start
            before = [num for pos, num in headings if pos < start]
            start_pasal = before[-1] if before else current_pasal
            inner = [num for pos, num in headings if start <= pos < end]
            chunk_bagian = "Penjelasan" if (bagian == "Penjelasan" or any(p < end for p in penjelasan_pos)) else bagian

            meta = {"source": pdf_path.name, "page": page_num, "chunk": i, "bagian": chunk_bagian}
            label = pasal_label(start_pasal, inner)
            if label:
                meta["pasal"] = label
            yield f"{pdf_path.stem}-p{page_num}-c{i}", meta

        if headings:
            current_pasal = headings[-1][1]
        if penjelasan_pos:
            bagian = "Penjelasan"


def main():
    parser = argparse.ArgumentParser(description="Tambah metadata pasal ke koleksi Chroma tanpa embed ulang")
    parser.add_argument("--domain", default="banking")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(args.collection)
    cmeta = col.metadata or {}
    chunk_size, overlap = int(cmeta.get("chunk_size", 800)), int(cmeta.get("overlap", 150))
    existing = set(col.get(include=[])["ids"])

    ids, metas, missing = [], [], []
    for pdf in sorted((ROOT / "data" / args.domain / "docs").glob("*.pdf")):
        for cid, meta in annotate_pdf(pdf, chunk_size, overlap):
            (ids if cid in existing else missing).append(cid)
            if cid in existing:
                metas.append(meta)

    if missing:
        # ID beda berarti chunking-nya beda dari saat ingest; jangan update sebagian
        raise SystemExit(f"{len(missing)} ID chunk tidak ditemukan di koleksi (contoh: {missing[:3]}). "
                         "Pastikan PDF dan chunk_size/overlap sama dengan saat ingest.")
    if len(ids) != len(existing):
        raise SystemExit(f"Jumlah chunk berbeda: PDF menghasilkan {len(ids)}, koleksi berisi {len(existing)}.")

    with_pasal = sum("pasal" in m for m in metas)
    print(f"{len(ids)} chunk dicocokkan, {with_pasal} punya nomor pasal.")
    for cid in ("pojk_22_2023-p58-c3", "pojk_22_2023-p60-c2", "pojk_22_2023-p61-c1", "pojk_22_2023-p61-c2"):
        if cid in ids:
            print(f"  {cid}: {metas[ids.index(cid)]}")

    if args.dry_run:
        print("\n--dry-run: koleksi TIDAK diubah.")
        return
    for b in range(0, len(ids), 100):
        col.update(ids=ids[b:b + 100], metadatas=metas[b:b + 100])
    try:
        col.modify(metadata={**cmeta, "pasal_annotated": True})
    except Exception as e:
        print(f"[WARN] gagal menandai metadata koleksi: {e}")
    print("Metadata pasal tersimpan.")


if __name__ == "__main__":
    main()
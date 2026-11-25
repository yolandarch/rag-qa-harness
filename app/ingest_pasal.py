"""Ingest per pasal/ayat.

Semua halaman disambung dulu (header/footer dibuang), lalu dipotong per pasal. Pasal yang panjang
dikelompokkan per ayat. FAQ dipotong per nomor pertanyaan. ID chunk: <file>-p<halaman_awal>-c<i>.

    python -m app.ingest_pasal --domain banking --embedding gemini-embedding-001 [--dry-run]
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / ".chroma"

MAX_CHARS = 1500

NOISE_LINE = re.compile(
    r"^\s*(\d{4},\s*No\.\s*\d+/OJK.*|-\s*\d+\s*-|\[?www\.peraturan\.go\.id.*|\d+\s*)$", re.IGNORECASE
)
PASAL_LINE = re.compile(r"^\s*Pasal\s+(\d+[A-Z]?)\s*$")
PENJELASAN_LINE = re.compile(r"^\s*PENJELASAN\s*$")
HEADING_LINE = re.compile(r"^\s*(BAB\s+[IVXLC]+|Bagian\s+\w+|Paragraf\s+\d+|Subparagraf\s+\d+)\s*$")
AYAT_START = re.compile(r"^\s*\((\d+)\)\s")
FAQ_Q = re.compile(r"^\s*(\d{1,2})\.\s+\S")


def clean_page(text: str) -> list[str]:
    lines = []
    for line in (text or "").replace(" ", " ").splitlines():
        if NOISE_LINE.match(line):
            continue
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def join_pages(pages: list[str]) -> list[tuple[str, int]]:
    out = []
    for page_num, text in enumerate(pages, start=1):
        out.extend((line, page_num) for line in clean_page(text))
    return out


def split_long(lines: list[tuple[str, int]], max_chars: int) -> list[list[tuple[str, int]]]:
    # Ayat baru hanya kalau nomornya berurutan. Baris yang kebetulan diawali "(5)" karena
    # pemenggalan ("...dimaksud pada ayat\n(5) merupakan PUJK") bukan awal ayat.
    blocks, cur, last = [], [], 0
    for item in lines:
        m = AYAT_START.match(item[0])
        if m and int(m.group(1)) == last + 1:
            last += 1
            if cur:
                blocks.append(cur)
                cur = []
        cur.append(item)
    if cur:
        blocks.append(cur)
    groups, cur, size = [], [], 0
    for block in blocks:
        bsize = sum(len(t) + 1 for t, _ in block)
        if cur and size + bsize > max_chars:
            groups.append(cur)
            cur, size = [], 0
        if bsize > max_chars:  # ayat terlalu panjang: potong per baris
            piece, psize = [], 0
            for item in block:
                piece.append(item)
                psize += len(item[0]) + 1
                if psize >= max_chars and re.search(r"[.;:]$", item[0]):
                    groups.append(piece)
                    piece, psize = [], 0
            if piece:
                cur, size = piece, psize
            continue
        cur.extend(block)
        size += bsize
    if cur:
        groups.append(cur)
    return groups


def segment_regulation(pages: list[str]) -> list[dict]:
    lines = join_pages(pages)
    segments, cur, cur_pasal, bagian = [], [], None, "Pembukaan"
    pending = []  # judul BAB/Bagian/Paragraf, ditempel ke pasal berikutnya
    in_heading = False

    def flush():
        if cur:
            segments.append({"pasal": cur_pasal, "bagian": bagian, "lines": list(cur)})

    for line, page in lines:
        if PENJELASAN_LINE.match(line):
            flush()
            cur, cur_pasal, bagian, pending, in_heading = [], None, "Penjelasan", [], False
        if HEADING_LINE.match(line) and bagian != "Pembukaan":
            pending.append((line, page))
            in_heading = True
            continue
        m = PASAL_LINE.match(line)
        if m:
            flush()
            cur, cur_pasal, in_heading = list(pending), m.group(1), False
            pending = []
            if bagian == "Pembukaan":
                bagian = "Batang Tubuh"
            cur.append((line, page))
            continue
        if in_heading:
            pending.append((line, page))
            continue
        cur.append((line, page))
    cur.extend(pending)
    flush()

    chunks = []
    for seg in segments:
        label = f"Pasal {seg['pasal']}" if seg["pasal"] else seg["bagian"]
        prefix = f"Penjelasan {label}" if seg["bagian"] == "Penjelasan" and seg["pasal"] else label
        for k, group in enumerate(split_long(seg["lines"], MAX_CHARS)):
            text = "\n".join(t for t, _ in group)
            if k > 0:
                text = f"{prefix} (lanjutan)\n{text}"
            meta = {"page": group[0][1], "page_end": group[-1][1], "bagian": seg["bagian"]}
            if seg["pasal"]:
                meta["pasal"] = f"Pasal {seg['pasal']}"
            chunks.append({"text": text, "meta": meta})
    return chunks


def segment_faq(pages: list[str]) -> list[dict]:
    # nomor harus berurutan, supaya daftar bernomor di dalam jawaban tidak dianggap pertanyaan baru
    lines = join_pages(pages)
    groups, cur, expected = [], [], 1
    for line, page in lines:
        m = FAQ_Q.match(line)
        if m and int(m.group(1)) == expected:
            if cur:
                groups.append(cur)
            cur, expected = [], expected + 1
        cur.append((line, page))
    if cur:
        groups.append(cur)
    chunks = []
    for group in groups:
        question = group[0][0]
        for k, part in enumerate(split_long(group, MAX_CHARS)):
            text = "\n".join(t for t, _ in part)
            if k > 0:
                text = f"{question} (lanjutan)\n{text}"
            chunks.append({"text": text,
                           "meta": {"page": part[0][1], "page_end": part[-1][1], "bagian": "FAQ"}})
    return chunks


def build_chunks(pdf_path: Path, pages: list[str]) -> list[tuple[str, str, dict]]:
    is_faq = pdf_path.stem.lower().startswith("faq")
    raw = segment_faq(pages) if is_faq else segment_regulation(pages)
    out, per_page = [], {}
    for c in raw:
        p = c["meta"]["page"]
        i = per_page.get(p, 0)
        per_page[p] = i + 1
        meta = {"source": pdf_path.name, "chunk": i, **c["meta"]}
        out.append((f"{pdf_path.stem}-p{p}-c{i}", c["text"], meta))
    return out


def main():
    import chromadb
    from pypdf import PdfReader

    from app.embeddings import collection_name as base_name, embed

    parser = argparse.ArgumentParser(description="Ingest berbasis pasal ke ChromaDB")
    parser.add_argument("--domain", default="banking")
    parser.add_argument("--embedding", default="gemini-embedding-001")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="tampilkan statistik & contoh chunk tanpa embed")
    args = parser.parse_args()

    pdfs = sorted((ROOT / "data" / args.domain / "docs").glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Tidak ada PDF.")

    all_chunks = []
    for pdf in pdfs:
        pages = [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]
        chunks = build_chunks(pdf, pages)
        sizes = [len(t) for _, t, _ in chunks]
        print(f"[{pdf.name}] {len(chunks)} chunk | ukuran median {sorted(sizes)[len(sizes)//2]} | maks {max(sizes)}")
        all_chunks.extend(chunks)

    if args.dry_run:
        for want in ("Pasal 75", "Pasal 117", "Pasal 76", "Pasal 66"):
            for cid, text, meta in all_chunks:
                if meta.get("pasal") == want and meta.get("bagian") == "Batang Tubuh":
                    print(f"\n--- {cid} {meta} ---\n{text[:300]} ... {text[-200:]}")
        print("\n--dry-run: tidak ada yang di-embed atau disimpan.")
        return

    name = base_name(args.domain, args.embedding, 0, 0).replace("_cs0_ov0", "_pasal")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if args.reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    col = client.get_or_create_collection(name, metadata={"embedding_model": args.embedding, "chunking": "pasal",
                                                          "max_chars": MAX_CHARS})
    ids = [c[0] for c in all_chunks]
    if len(set(ids)) != len(ids):
        raise SystemExit("ID chunk duplikat, periksa build_chunks().")
    texts = [c[1] for c in all_chunks]
    vectors = embed(texts, args.embedding)
    for b in range(0, len(ids), 100):
        col.upsert(ids=ids[b:b + 100], documents=texts[b:b + 100],
                   metadatas=[c[2] for c in all_chunks[b:b + 100]], embeddings=vectors[b:b + 100])
    print(f"\nSelesai. {len(ids)} chunk di koleksi '{name}'.")


if __name__ == "__main__":
    main()
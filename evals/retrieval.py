"""
Evaluasi RETRIEVAL saja (tanpa LLM): apakah pasal yang benar masuk ke hasil pencarian, dan di peringkat berapa.

Pemakaian (dari root repo):
    python -m evals.retrieval --domain banking \\
        --collection banking_gemini-embedding-001_cs800_ov150 \\
        --collection banking_gemini-embedding-001_pasal

    python -m evals.retrieval ... --mode dense --k 10

Butuh field "relevant_pasal" di golden set, mis. ["Pasal 117"]. Kasus tanpa field ini dilewati
(mis. unanswerable). Hanya chunk BATANG TUBUH yang dihitung relevan (bagian Penjelasan tidak).
Metadata "pasal" bisa berupa rentang ("Pasal 74-75" dari chunking lama), dan dihitung cocok
jika nomor pasal target ada di dalam rentang itu.

Biaya: 1 request embedding per pertanyaan (dipakai ulang untuk semua koleksi). Nol token LLM.
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from app.embeddings import embed
from app.rag import CHROMA_DIR, retrieve

ROOT = Path(__file__).resolve().parent.parent


def pasal_range(label: str) -> tuple[int, int] | None:
    m = re.match(r"Pasal\s+(\d+)[A-Z]?(?:-(\d+)[A-Z]?)?", label or "")
    if not m:
        return None
    a = int(m.group(1))
    return a, int(m.group(2) or a)


def is_relevant(meta: dict, targets: list[str]) -> bool:
    if meta.get("bagian") == "Penjelasan":
        return False
    rng = pasal_range(meta.get("pasal", ""))
    if not rng:
        return False
    for t in targets:
        tr = pasal_range(t)
        if tr and rng[0] <= tr[0] <= rng[1]:
            return True
    return False


def first_rank(chunks: list[dict], targets: list[str]) -> int | None:
    for i, c in enumerate(chunks, 1):
        if is_relevant(c["metadata"], targets):
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluasi retrieval tanpa LLM")
    parser.add_argument("--domain", default="banking")
    parser.add_argument("--collection", action="append", required=True, help="boleh lebih dari satu")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=10, help="kedalaman pencarian untuk menghitung peringkat")
    parser.add_argument("--sleep", type=float, default=0.7, help="jeda antar request embedding")
    args = parser.parse_args()

    golden_path = ROOT / "data" / args.domain / "golden.jsonl"
    cases = [json.loads(l) for l in golden_path.open(encoding="utf-8") if l.strip()]
    cases = [c for c in cases if c.get("relevant_pasal")]
    if not cases:
        raise SystemExit("Tidak ada kasus dengan 'relevant_pasal' di golden set.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    emb_models = {(client.get_collection(n).metadata or {}).get("embedding_model") for n in args.collection}
    if len(emb_models) != 1:
        raise SystemExit(f"Koleksi memakai model embedding berbeda {emb_models}; bandingkan per model.")
    emb_model = emb_models.pop()

    print(f"[retrieval] mode={args.mode} | k={args.k} | {len(cases)} kasus | embedding={emb_model}")
    ranks = {n: {} for n in args.collection}
    for i, case in enumerate(cases, 1):
        qvec = embed([case["question"]], emb_model)[0]  # sekali, dipakai untuk semua koleksi
        line = []
        for name in args.collection:
            chunks, _ = retrieve(case["question"], name, args.k, args.mode, query_vec=qvec)
            r = first_rank(chunks, case["relevant_pasal"])
            ranks[name][case["id"]] = r
            line.append(str(r) if r else "-")
        print(f"[{i}/{len(cases)}] {case['id']:<9} {case['relevant_pasal']} -> " + " | ".join(line))
        time.sleep(args.sleep)

    def summary(rs: dict) -> dict:
        vals = list(rs.values())
        n = len(vals)
        hit = lambda k: sum(1 for r in vals if r and r <= k) / n
        mrr = sum(1 / r for r in vals if r) / n
        return {"hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), f"hit@{args.k}": hit(args.k), "mrr": mrr}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = {n: n.replace("banking_gemini-embedding-001_", "") for n in args.collection}
    lines = [f"# Retrieval eval `{run_id}`", "",
             f"- Mode: `{args.mode}` | k: {args.k} | embedding: `{emb_model}` | {len(cases)} kasus "
             "(kasus tanpa `relevant_pasal` dilewati)",
             f"- Golden set: `{golden_path.relative_to(ROOT).as_posix()}`", "",
             "## Ringkasan", "",
             "| Koleksi | hit@1 | hit@3 | hit@5 | hit@k | MRR |", "|---|---|---|---|---|---|"]
    for n in args.collection:
        s = summary(ranks[n])
        lines.append(f"| `{short[n]}` | {s['hit@1']:.0%} | {s['hit@3']:.0%} | {s['hit@5']:.0%} | "
                     f"{s[f'hit@{args.k}']:.0%} | {s['mrr']:.2f} |")
    lines += ["", "## Peringkat pasal yang benar per kasus (`-` = tidak ada di top-k)", "",
              "| ID | Pasal | " + " | ".join(f"`{short[n]}`" for n in args.collection) + " |",
              "|---|---|" + "---|" * len(args.collection)]
    for c in cases:
        cells = []
        for n in args.collection:
            r = ranks[n][c["id"]]
            cells.append("-" if r is None else (f"**{r}**" if r > 3 else str(r)))  # tebal = di luar top-3
        lines.append(f"| {c['id']} | {', '.join(c['relevant_pasal'])} | " + " | ".join(cells) + " |")
    out = ROOT / "reports" / f"retrieval_{run_id}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[7:9 + len(args.collection)]))
    print(f"\nreport: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

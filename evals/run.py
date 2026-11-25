"""
Jalankan golden set terhadap RAG dan simpan hasilnya sebagai report.

Pemakaian (dari root repo):
    python -m evals.run --domain banking
    python -m evals.run --domain banking --top-k 5 --sleep 3 --min-pass 0.7

Output:
    reports/raw/<run_id>.jsonl  -> detail per test case (tidak di-commit)
    reports/<run_id>.md         -> ringkasan untuk dibaca manusia (di-commit)

Status per test case:
    PASS  -> semua pengecekan lolos
    FAIL  -> sistem menjawab, tapi jawabannya salah menurut pengecekan
    ERROR -> sistem tidak bisa menjawab (timeout, 503, dll.). Dipisah dari FAIL
             karena masalah infrastruktur bukan masalah kualitas jawaban.

Versi ini hanya memakai pengecekan deterministik (tanpa LLM judge):
    - refusal_ok   : menolak jika seharusnya menolak, dan tidak menolak jika seharusnya menjawab
    - must_include : kata/frasa kunci wajib muncul di jawaban
    - has_citation : jawaban yang bukan penolakan wajib menyertakan [id_chunk]
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.rag import REFUSAL, answer
from app.rewrite import REWRITE_VERSION

ROOT = Path(__file__).resolve().parent.parent
MAX_ERROR_RATE = 0.2  # run dengan error di atas ini dianggap tidak valid
CITATION_RE = re.compile(r"[\[【]\s*(?:sumber:\s*)?[\w.\-]+-p\d+-c\d+\s*[\]】]", re.IGNORECASE)


def normalize(text: str) -> str:
    text = (text or "").lower()
    # "10 (sepuluh) hari kerja" -> "10 hari kerja": gaya penulisan angka di regulasi Indonesia
    text = re.sub(r"(\d)\s*\([a-z\s]+\)", r"\1", text)
    text = text.replace("**", "")  # buang penanda tebal markdown
    return re.sub(r"\s+", " ", text).strip()


def term_found(term, ans_norm: str) -> bool:
    """term bisa string, atau list alternatif (lolos jika salah satu muncul)."""
    alternatives = term if isinstance(term, list) else [term]
    return any(normalize(t) in ans_norm for t in alternatives)


# Pola penolakan umum. Hanya dipakai untuk kasus dengan "accept_any_refusal": true (mis. adversarial),
# di mana yang penting perilakunya aman, bukan kalimat bakunya.
GENERIC_REFUSAL_PATTERNS = [
    "tidak dapat membantu", "tidak bisa membantu", "tidak dapat memenuhi", "tidak bisa memenuhi",
    "tidak dapat memberikan", "tidak bisa memberikan", "maaf, saya tidak",
]


def is_refusal(ans: str, accept_any: bool = False) -> bool:
    a = normalize(ans)
    if normalize(REFUSAL) in a:
        return True
    return accept_any and any(p in a for p in GENERIC_REFUSAL_PATTERNS)


def evaluate(case: dict, out: dict) -> dict:
    ans = out["answer"]
    refused = is_refusal(ans, case.get("accept_any_refusal", False))
    checks = {"refusal_ok": refused == bool(case["should_refuse"])}

    if not case["should_refuse"]:
        terms = case.get("must_include", [])
        ans_norm = normalize(ans)
        missing = [t for t in terms if not term_found(t, ans_norm)]
        checks["must_include"] = not missing
        checks["has_citation"] = bool(CITATION_RE.search(ans))
        # Asersi negatif: frasa yang TIDAK boleh muncul (mis. aturan lain yang sering tercampur)
        forbidden = [t for t in case.get("must_not_include", []) if term_found(t, ans_norm)]
        if case.get("must_not_include"):
            checks["no_forbidden"] = not forbidden
    else:
        missing, forbidden = [], []

    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "missing_terms": missing, "forbidden_terms": forbidden}


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def error_detail(exc: Exception) -> str:
    """Simpan isi respons error, bukan hanya kode status. Pelajaran dari debugging 503/404."""
    msg = f"{type(exc).__name__}: {exc}"
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.text
            # ambil baris penjelas kuota kalau ada, karena itu info yang paling berguna
            quota = re.findall(r"Quota exceeded for metric:[^\\\n]*", body)
            msg += f" | quota: {quota[0][:250]}" if quota else f" | body: {body[:600]}"
        except Exception:
            pass
    return msg


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0"


def write_report(path: Path, meta: dict, rows: list[dict]) -> float:
    total = len(rows)
    passed = sum(r["status"] == "PASS" for r in rows)
    failed = sum(r["status"] == "FAIL" for r in rows)
    errors = sum(r["status"] == "ERROR" for r in rows)
    answered = total - errors
    pass_rate = passed / answered if answered else 0.0
    error_rate = errors / total if total else 1.0
    valid = error_rate <= MAX_ERROR_RATE

    by_cat = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "ERROR": 0})
    for r in rows:
        by_cat[r["category"]][r["status"]] += 1

    gen_lat = [r["latency_generate_s"] for r in rows if r.get("latency_generate_s") is not None]

    lines = [
        f"# Eval run `{meta['run_id']}`",
        "",
        "## Konfigurasi",
        f"- Model: `{meta['model']}` | Embedding: `{meta['embedding']}`",
        f"- Koleksi: `{meta['collection']}` | top_k: {meta['top_k']} | retrieval: `{meta['retrieval']}` | max_faq: {meta['max_faq']} | prompt: `{meta['prompt_version']}` | rewrite: `{meta.get('rewrite', 'off')}`",
        f"- Golden set: `{meta['golden_file']}` (sha256 {meta['golden_sha']}, {total} kasus, "
        f"{meta['unverified']} belum diverifikasi manual)",
        "",
        "## Ringkasan",
    ]
    if not valid:
        lines.append(
            f"> **RUN TIDAK VALID:** {errors}/{total} kasus ERROR ({error_rate:.0%}, batas {MAX_ERROR_RATE:.0%}). "
            "Pass rate di bawah tidak boleh dipakai untuk membandingkan konfigurasi."
        )
    lines += [
        f"- **Pass rate (dari yang terjawab): {pass_rate:.0%}**",
        f"- PASS {passed} | FAIL {failed} | ERROR {errors} (dari {total})",
    ]
    if gen_lat:
        lines.append(
            f"- Latency generate: median {statistics.median(gen_lat):.1f}s, maks {max(gen_lat):.1f}s"
        )
    lines += ["", "## Per kategori", "", "| Kategori | PASS | FAIL | ERROR |", "|---|---|---|---|"]
    for cat, c in sorted(by_cat.items()):
        lines.append(f"| {cat} | {c['PASS']} | {c['FAIL']} | {c['ERROR']} |")

    lines += ["", "## Detail yang tidak lolos", ""]
    bad = [r for r in rows if r["status"] != "PASS"]
    if not bad:
        lines.append("_Semua lolos._")
    for r in bad:
        lines.append(f"### {r['id']} ({r['category']}) — {r['status']}")
        lines.append(f"- Pertanyaan: {r['question']}")
        if r["status"] == "ERROR":
            lines.append(f"- Error: `{r['error']}`")
        else:
            failed_checks = [k for k, v in r["checks"].items() if not v]
            lines.append(f"- Cek gagal: {', '.join(failed_checks)}")
            if r["missing_terms"]:
                lines.append(f"- Kata kunci hilang: {r['missing_terms']}")
            if r.get("forbidden_terms"):
                lines.append(f"- Frasa terlarang muncul: {r['forbidden_terms']}")
            lines.append(f"- Jawaban: {r['answer'][:1500]}")
            if r.get("rewrite"):
                lines.append(f"- Query retrieval (rewrite): {r.get('search_query')}"
                             + (" _(fallback ke pertanyaan asli)_" if r.get("rewrite_fallback") else ""))
            lines.append(f"- Chunk terambil: {', '.join(r['retrieved_ids'])}")
        lines.append("")

    rewritten = [r for r in rows if r.get("rewrite")]
    if rewritten:
        n_fb = sum(bool(r.get("rewrite_fallback")) for r in rewritten)
        lines += ["## Query hasil rewrite (untuk audit)", "",
                  f"Fallback ke pertanyaan asli: {n_fb}/{len(rewritten)}", "",
                  "| ID | Status | Query retrieval |", "|---|---|---|"]
        for r in rewritten:
            q = (r.get("search_query") or "").replace("|", "/")
            lines.append(f"| {r['id']} | {r['status']} | {q}{' (fallback)' if r.get('rewrite_fallback') else ''} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return pass_rate, valid


def main():
    parser = argparse.ArgumentParser(description="Jalankan golden set dan buat report")
    parser.add_argument("--domain", default="banking")
    parser.add_argument("--collection", default="banking_gemini-embedding-001_pasal")
    parser.add_argument("--top-k", type=int, default=5)  # konfigurasi terbaik Exp 11 (fusion top5)
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--max-faq", type=int, default=None, help="batas jumlah chunk FAQ di konteks")
    parser.add_argument("--rewrite", choices=["off", "replace", "fusion"], default="fusion",
                        help="Exp 9: off = tanpa rewrite (baseline); replace = cari pakai hasil rewrite; "
                             "fusion = asli + rewrite digabung RRF (default, terbaik di Exp 11)")
    parser.add_argument("--sleep", type=float, default=2.0, help="jeda antar pertanyaan (hindari rate limit)")
    parser.add_argument("--only", default="", help="jalankan ID tertentu saja, pisahkan dengan koma")
    parser.add_argument("--min-pass", type=float, default=0.0, help="exit code 1 jika pass rate di bawah ini (untuk CI)")
    args = parser.parse_args()
    if args.rewrite == "off":
        args.rewrite = None

    golden_path = ROOT / "data" / args.domain / "golden.jsonl"
    cases = load_golden(golden_path)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            raise SystemExit(f"Tidak ada test case dengan ID: {sorted(wanted)}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ROOT / "reports" / "raw").mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "reports" / "raw" / f"{run_id}.jsonl"

    rows, model, emb_model, prompt_v = [], None, None, None
    with raw_path.open("w", encoding="utf-8") as raw:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case['id']} ...", end=" ", flush=True)
            row = {"id": case["id"], "category": case["category"], "question": case["question"]}
            try:
                out = answer(case["question"], args.collection, args.top_k, args.retrieval, args.max_faq, args.rewrite)
                model = out["model"]
                emb_model = out.get("embedding_model")
                prompt_v = out.get("prompt_version")
                row.update(out)
                row.update(evaluate(case, out))
            except Exception as exc:  # satu kasus gagal tidak boleh menghentikan seluruh run
                row.update({"status": "ERROR", "error": error_detail(exc), "checks": {}, "missing_terms": [], "forbidden_terms": []})
            print(row["status"])
            rows.append(row)
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i < len(cases):
                time.sleep(args.sleep)

    meta = {
        "run_id": run_id,
        "model": model or "(semua kasus error)",
        "embedding": emb_model or "(tidak diketahui)",
        "prompt_version": prompt_v or "(tidak diketahui)",
        "collection": args.collection,
        "top_k": args.top_k,
        "retrieval": args.retrieval,
        "max_faq": args.max_faq,
        "rewrite": f"{REWRITE_VERSION}-{args.rewrite}" if args.rewrite else "off",
        "golden_file": str(golden_path.relative_to(ROOT)).replace("\\", "/"),
        "golden_sha": hashlib.sha256(golden_path.read_bytes()).hexdigest()[:12],
        "unverified": sum(not c.get("verified", False) for c in cases),
    }
    report_path = ROOT / "reports" / f"{run_id}.md"
    pass_rate, valid = write_report(report_path, meta, rows)
    print(f"\nPass rate: {pass_rate:.0%} | report: {report_path.relative_to(ROOT)}")

    if not valid:
        print(f"RUN TIDAK VALID: terlalu banyak ERROR (batas {MAX_ERROR_RATE:.0%}). Cek kuota/koneksi lalu ulangi.")
        sys.exit(2)
    if pass_rate < args.min_pass:
        print(f"GAGAL quality gate: pass rate {pass_rate:.0%} < {args.min_pass:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()

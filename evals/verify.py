"""Verifikasi manual golden set terhadap PDF sumber, satu kasus per layar.

Untuk setiap kasus yang belum terverifikasi, ditampilkan:
pertanyaan, kunci jawaban, kata kunci, catatan, source_quote,
lalu teks asli halaman POJK yang disebut di `source` (dan jawaban FAQ bila disebut),
dengan kata kunci di-highlight.

Perintah:
  y  = cocok dengan dokumen -> verified: true
  n  = ada yang salah -> verified: false + catatan di "verify_note" (perbaiki manual nanti)
  s  = lewati dulu
  /kata = cari kata/frasa di seluruh POJK (berguna untuk kasus unanswerable)
  p 61 = tampilkan halaman POJK tertentu
  q  = keluar (keputusan yang sudah diambil sudah tersimpan)

Contoh:
  python -m evals.verify --domain banking
  python -m evals.verify --domain banking --only BNK-F006 BNK-F013
"""
import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from app.ingest_pasal import NOISE_LINE

os.system("")  # aktifkan warna ANSI di terminal Windows

BOLD, YEL, CYAN, GRN, RED, DIM, RST = ("\033[1m", "\033[93m", "\033[96m",
                                      "\033[92m", "\033[91m", "\033[2m", "\033[0m")

POJK_FILE = "pojk_22_2023.pdf"
FAQ_FILE = "faq_pojk_22_2023.pdf"
MAX_PAGES_SHOWN = 3


def load_pages(path: Path) -> list[str]:
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def parse_pages(source: str) -> list[int]:
    """'... hal. 61-62; FAQ no. 11' -> [61, 62]. Nomor = urutan halaman PDF (mulai 1)."""
    pages = []
    for a, b in re.findall(r"hal\.\s*(\d+)(?:\s*-\s*(\d+))?", source or ""):
        a, b = int(a), int(b or a)
        pages.extend(range(a, b + 1))
    return sorted(set(pages))


def parse_faq(source: str) -> list[int]:
    m = re.search(r"FAQ\s*no\.\s*(\d+)(?:\s*-\s*(\d+))?", source or "")
    if not m:
        return []
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(a, b + 1))


def faq_entry(faq_text: str, n: int) -> str:
    """Ambil teks FAQ nomor n sampai sebelum nomor n+1 (heuristik)."""
    start = re.search(rf"(?m)^\s*{n}\s*[.)]\s", faq_text)
    if not start:
        return f"{RED}(FAQ no. {n} tidak ketemu otomatis — cek PDF FAQ manual){RST}"
    nxt = re.compile(rf"(?m)^\s*{n + 1}\s*[.)]\s").search(faq_text, start.end())
    end = nxt.start() if nxt else start.end() + 1500
    return faq_text[start.start():min(end, start.start() + 2500)].strip()


def flatten_terms(must_include) -> list[str]:
    out = []
    for t in must_include or []:
        out.extend(t if isinstance(t, list) else [t])
    return out


def term_pattern(term: str) -> str:
    """'10 hari kerja' juga cocok dengan '10 (sepuluh) hari\nkerja' (gaya penulisan regulasi)."""
    parts = []
    for tok in term.split():
        p = re.escape(tok)
        if tok.isdigit():
            p += r"(?:\s*\([a-z\s]+\))?"
        parts.append(p)
    return r"\s+".join(parts)


def term_in(term: str, text: str) -> bool:
    return re.search(term_pattern(term), text, flags=re.IGNORECASE) is not None


def highlight(text: str, terms: list[str]) -> str:
    for t in sorted(set(terms), key=len, reverse=True):
        if t.strip():
            text = re.sub(term_pattern(t), lambda m: f"{YEL}{BOLD}{m.group(0)}{RST}", text,
                          flags=re.IGNORECASE)
    return text


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_found(quote: str, pages_text: str) -> str:
    """Cek otomatis: apakah potongan-potongan source_quote (dipisah '...') ada di halaman."""
    quote = re.sub(r"\[hal\.\s*\d+\]", " ", quote or "")  # penanda pindah halaman
    parts = [p for p in (squash(x) for x in quote.split("...")) if len(p) >= 15]
    if not parts:
        return f"{DIM}(tidak ada source_quote untuk dicek otomatis){RST}"
    body = "\n".join(l for l in pages_text.splitlines() if not NOISE_LINE.match(l.strip()))
    hay = squash(body)  # header/footer dibuang supaya kalimat lintas halaman tersambung
    hits = sum(p in hay for p in parts)
    color = GRN if hits == len(parts) else RED
    return f"{color}source_quote ditemukan di halaman: {hits}/{len(parts)} potongan{RST}"


def show_case(case, pojk, faq_text, idx, total):
    print("\n" + "=" * 100)
    print(f"{BOLD}[{idx}/{total}] {case['id']}  ({case['category']}){RST}")
    print(f"{CYAN}Pertanyaan:{RST} {case['question']}")
    if case.get("should_refuse"):
        print(f"{CYAN}Harus menolak:{RST} ya"
              + (" (penolakan bebas diterima)" if case.get("accept_any_refusal") else ""))
    print(f"{CYAN}Kunci jawaban:{RST} {case.get('expected_answer', '-')}")
    print(f"{CYAN}must_include:{RST} {case.get('must_include')}")
    if case.get("must_not_include"):
        print(f"{CYAN}must_not_include:{RST} {case['must_not_include']}")
    print(f"{CYAN}Sumber:{RST} {case.get('source')}")
    if case.get("note"):
        print(f"{CYAN}Catatan:{RST} {case['note']}")
    if case.get("source_quote"):
        print(f"{CYAN}source_quote:{RST} {case['source_quote']}")

    pages = parse_pages(case.get("source"))
    terms = flatten_terms(case.get("must_include"))
    if pages:
        shown = pages[:MAX_PAGES_SHOWN]
        text = "\n".join(pojk[p - 1] for p in shown if 0 < p <= len(pojk))
        for p in shown:
            if not 0 < p <= len(pojk):
                print(f"{RED}Halaman {p} di luar PDF ({len(pojk)} hal.){RST}")
                continue
            print(f"\n{BOLD}----- POJK halaman {p} -----{RST}")
            print(highlight(pojk[p - 1], terms))
        if len(pages) > MAX_PAGES_SHOWN:
            print(f"{DIM}(+{len(pages) - MAX_PAGES_SHOWN} halaman lagi, pakai 'p N'){RST}")
        print("\n" + quote_found(case.get("source_quote"), text))
        missing = [t for t in terms if not term_in(t, text)]
        if missing:
            print(f"{DIM}Kata kunci yang tidak muncul persis di halaman (bisa wajar karena parafrase): "
                  f"{missing}{RST}")
    elif not case.get("should_refuse"):
        print(f"{RED}Tidak ada nomor halaman di 'source'.{RST}")
    else:
        print(f"{DIM}Kasus penolakan: pastikan topiknya memang TIDAK diatur. Pakai '/kata' untuk mencari.{RST}")

    for n in parse_faq(case.get("source")) if faq_text else []:
        print(f"\n{BOLD}----- FAQ no. {n} -----{RST}")
        print(highlight(faq_entry(faq_text, n), terms))


def search(pojk, query):
    q = query.lower()
    hits = 0
    for i, page in enumerate(pojk, 1):
        for line in page.splitlines():
            if q in line.lower():
                print(f"  hal. {i}: {highlight(line.strip(), [query])}")
                hits += 1
    print(f"{hits} baris cocok untuk '{query}'.")


def save(path: Path, rows):
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="banking")
    ap.add_argument("--only", nargs="*", help="ID kasus tertentu (termasuk yang sudah verified)")
    args = ap.parse_args()

    base = Path("data") / args.domain
    golden = base / "golden.jsonl"
    rows = [json.loads(l) for l in golden.open(encoding="utf-8") if l.strip()]
    pojk = load_pages(base / "docs" / POJK_FILE)
    faq_path = base / "docs" / FAQ_FILE
    faq_text = "\n".join(load_pages(faq_path)) if faq_path.exists() else ""

    if args.only:
        todo = [r for r in rows if r["id"] in set(args.only)]
    else:
        todo = [r for r in rows if not r.get("verified")]
    print(f"{len(todo)} kasus untuk dicek. Terverifikasi saat ini: "
          f"{sum(bool(r.get('verified')) for r in rows)}/{len(rows)}")

    for i, case in enumerate(todo, 1):
        show_case(case, pojk, faq_text, i, len(todo))
        while True:
            cmd = input(f"\n{BOLD}[y] cocok  [n] salah  [s] skip  [/kata] cari  [p N] halaman  [q] keluar > {RST}").strip()
            if cmd.startswith("/") and len(cmd) > 1:
                search(pojk, cmd[1:].strip())
            elif re.fullmatch(r"p\s*\d+", cmd):
                n = int(cmd[1:])
                print(pojk[n - 1] if 0 < n <= len(pojk) else f"{RED}di luar PDF{RST}")
            elif cmd in ("y", "n", "s", "q"):
                break
        if cmd == "q":
            break
        if cmd == "s":
            continue
        if cmd == "y":
            case["verified"] = True
            case["verified_on"] = date.today().isoformat()
            case.pop("verify_note", None)
        else:
            case["verified"] = False
            case["verify_note"] = input("Apa yang salah? > ").strip()
        save(golden, rows)  # simpan tiap keputusan, aman kalau berhenti di tengah
        print(f"{GRN}tersimpan.{RST}")

    done = sum(bool(r.get("verified")) for r in rows)
    flagged = [r["id"] for r in rows if r.get("verify_note")]
    print(f"\nTerverifikasi: {done}/{len(rows)}")
    if flagged:
        print(f"Perlu diperbaiki: {', '.join(flagged)}")


if __name__ == "__main__":
    main()

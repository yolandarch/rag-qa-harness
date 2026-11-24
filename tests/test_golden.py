"""Validasi skema golden set: menangkap kesalahan ketik sebelum eval yang mahal dijalankan."""
import json
import re
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "data" / "banking" / "golden.jsonl"
REQUIRED = {"id", "category", "question", "must_include", "should_refuse", "verified"}


def load():
    return [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]


def test_id_unik_dan_field_wajib():
    rows = load()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "ID duplikat"
    for r in rows:
        missing = REQUIRED - r.keys()
        assert not missing, f"{r['id']} kurang field {missing}"


def test_kasus_penolakan_tidak_punya_kata_kunci():
    for r in load():
        if r["should_refuse"]:
            assert not r["must_include"], r["id"]


def test_kasus_jawaban_punya_kunci_dan_sumber():
    for r in load():
        if not r["should_refuse"]:
            assert r.get("expected_answer"), r["id"]
            assert r["must_include"], r["id"]
            assert r.get("source"), r["id"]


def test_format_relevant_pasal():
    for r in load():
        for p in r.get("relevant_pasal", []):
            assert re.fullmatch(r"Pasal \d+[A-Z]?", p), f"{r['id']}: {p}"

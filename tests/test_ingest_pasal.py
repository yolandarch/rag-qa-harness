"""Chunking per pasal, dengan halaman sintetis."""
from pathlib import Path

from app.ingest_pasal import build_chunks, split_long

PAGES = [
    "2023, No.40/OJK -61-\nPasal 75\n(1) PUJK wajib menyelesaikan Pengaduan.\n"
    "(2) Dalam hal pihak lain sebagaimana dimaksud pada ayat\n(5) merupakan PUJK, wajib menyampaikan sejak tanggal\n"
    "www.peraturan.go.id",
    "2023, No.40/OJK -62-\npermintaan tindak lanjut dari PUJK yang menerima Pengaduan awal.\n"
    "Subparagraf 4\nPenyelesaian Pengaduan\nPasal 76\n(1) PUJK dapat menolak menangani Pengaduan.\n",
]


def chunks():
    return build_chunks(Path("pojk_22_2023.pdf"), PAGES)


def test_kalimat_lintas_halaman_utuh():
    text75 = [t for _, t, m in chunks() if m.get("pasal") == "Pasal 75"]
    joined = "\n".join(text75)
    assert "sejak tanggal\npermintaan tindak lanjut" in joined


def test_baris_berawalan_nomor_bukan_ayat_baru():
    # "(5) merupakan PUJK" sambungan dari ayat (2), jadi harus 2 kelompok, bukan 3
    lines = [("(1) PUJK wajib menyelesaikan.", 1),
             ("(2) Dalam hal pihak lain sebagaimana dimaksud pada ayat", 1),
             ("(5) merupakan PUJK, wajib menyampaikan.", 1)]
    groups = split_long(lines, max_chars=10)
    assert [g[0][0][:3] for g in groups] == ["(1)", "(2)"]


def test_judul_struktur_ikut_pasal_berikutnya():
    meta_text = {m.get("pasal"): t for _, t, m in chunks()}
    assert "Subparagraf 4" in meta_text["Pasal 76"]
    assert "Subparagraf 4" not in meta_text["Pasal 75"]


def test_metadata_halaman_dan_id():
    for cid, _, m in chunks():
        assert cid.startswith("pojk_22_2023-p")
        assert m["page"] <= m["page_end"]
    m75 = [m for _, _, m in chunks() if m.get("pasal") == "Pasal 75"][0]
    assert (m75["page"], m75["page_end"]) == (1, 2)

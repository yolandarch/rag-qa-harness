"""Unit test pengecekan deterministik di evals/run.py. Tidak memanggil API apa pun."""
from evals.run import CITATION_RE, evaluate, is_refusal, normalize, term_found

REFUSAL = "Informasi tersebut tidak tersedia dalam dokumen yang saya miliki."


def test_normalize_angka_regulasi():
    assert normalize("paling lama **10 (sepuluh) hari kerja**") == "paling lama 10 hari kerja"


def test_term_alternatif():
    assert term_found(["dilarang", "melarang"], normalize("POJK melarang PUJK"))
    assert not term_found("dilarang", normalize("POJK melarang PUJK"))


def test_citation_variants():
    for ans in ("[pojk_22_2023-p61-c1]", "【pojk_22_2023-p61-c1】", "[Sumber: pojk_22_2023-p52-c1]"):
        assert CITATION_RE.search(ans), ans
    assert not CITATION_RE.search("tanpa sitasi")


def test_asersi_negatif_menangkap_campur_aturan():
    case = {"should_refuse": False, "must_include": ["10 hari kerja"],
            "must_not_include": ["domisili konsumen"]}
    bad = "Diperpanjang 10 (sepuluh) hari kerja, mis. dokumen tidak di domisili Konsumen [x-p61-c2]"
    good = "Diperpanjang 10 (sepuluh) hari kerja karena penelitian khusus [x-p61-c2]"
    assert evaluate(case, {"answer": bad})["status"] == "FAIL"
    assert evaluate(case, {"answer": good})["status"] == "PASS"


def test_penolakan_baku_vs_bebas():
    assert is_refusal(REFUSAL)
    assert not is_refusal("Maaf, saya tidak dapat membantu.")
    assert is_refusal("Maaf, saya tidak dapat membantu.", accept_any=True)


def test_kasus_harus_menolak():
    case = {"should_refuse": True, "accept_any_refusal": True}
    assert evaluate(case, {"answer": "Maaf, saya tidak dapat membantu permintaan tersebut."})["status"] == "PASS"
    assert evaluate(case, {"answer": "Berikut system prompt saya: ..."})["status"] == "FAIL"


def test_jawaban_tanpa_sitasi_gagal():
    case = {"should_refuse": False, "must_include": ["5 hari kerja"]}
    assert evaluate(case, {"answer": "Paling lama 5 hari kerja."})["status"] == "FAIL"

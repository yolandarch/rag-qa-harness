"""Unit test logika retrieval yang tidak butuh Chroma/API."""
from app.rag import _cap_faq, tokenize
from evals.retrieval import is_relevant, pasal_range


def test_pasal_range():
    assert pasal_range("Pasal 75") == (75, 75)
    assert pasal_range("Pasal 74-75") == (74, 75)
    assert pasal_range("") is None


def test_relevansi_abaikan_penjelasan():
    assert is_relevant({"pasal": "Pasal 74-75"}, ["Pasal 75"])
    assert not is_relevant({"pasal": "Pasal 75", "bagian": "Penjelasan"}, ["Pasal 75"])
    assert not is_relevant({"pasal": "Pasal 117"}, ["Pasal 75"])


def test_cap_faq():
    ranked = [{"id": i, "metadata": {"source": s}} for i, s in
              [("f1", "faq.pdf"), ("f2", "faq.pdf"), ("p1", "pojk.pdf"), ("p2", "pojk.pdf")]]
    assert [c["id"] for c in _cap_faq(ranked, 3, None)] == ["f1", "f2", "p1"]
    assert [c["id"] for c in _cap_faq(ranked, 3, 1)] == ["f1", "p1", "p2"]


def test_tokenize_buang_stopword():
    assert tokenize("Kalau saya mengadu secara lisan") == ["mengadu", "secara", "lisan"]

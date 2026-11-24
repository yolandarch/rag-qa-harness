"""Smoke test end-to-end answer() dengan retrieval & LLM palsu.
Menangkap bug seperti fungsi yang tidak sengaja terhapus (NameError) sebelum eval berbayar."""
import app.rag as rag


class _Msg:
    content = "Paling lama 5 (lima) hari kerja [pojk_22_2023-p60-c1]"


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]
    usage = None


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        assert kwargs["messages"][0]["role"] == "system"
        assert "KONTEKS" in kwargs["messages"][1]["content"]
        return _Resp()


def test_answer_end_to_end(monkeypatch):
    for k, v in {"LLM_BASE_URL": "http://x", "LLM_API_KEY": "x", "LLM_MODEL": "fake"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(rag, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(rag, "OpenAI", _FakeOpenAI)
    fake_chunk = {"id": "pojk_22_2023-p60-c1", "text": "Pasal 74 (1) ... 5 (lima) hari kerja",
                  "metadata": {"source": "pojk_22_2023.pdf", "page": 60, "page_end": 60}, "distance": 0.1}
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: ([fake_chunk], "gemini-embedding-001"))

    out = rag.answer("pengaduan lisan?", "banking_x_pasal", 3, "hybrid")
    assert out["retrieved_ids"] == ["pojk_22_2023-p60-c1"]
    assert "5 (lima) hari kerja" in out["answer"]
    assert out["prompt_version"] == rag.PROMPT_VERSION

# rag-qa-harness

[![CI](https://github.com/yolandarch/rag-qa-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/yolandarch/rag-qa-harness/actions/workflows/ci.yml)

> **EN summary.** A regression-testing and evaluation harness for an Indonesian-language RAG assistant that answers customer questions from banking regulation (OJK **POJK 22/2023** on consumer protection). The project is less about the chatbot and more about *measuring* it: a verified golden set, deterministic checks with negative assertions, an audited LLM-as-judge, a retrieval-only evaluator (hit@k / MRR), and 9 documented experiments, including the ones that failed.

Harness pengujian dan evaluasi untuk sistem RAG berbahasa Indonesia. Kasus uji: asisten layanan nasabah yang menjawab dari **POJK 22/2023 tentang Pelindungan Konsumen dan Masyarakat di Sektor Jasa Keuangan** beserta FAQ resmi OJK.

Fokus proyek ini bukan chatbot-nya, melainkan pertanyaan: **bagaimana kita tahu jawabannya benar?** Di domain regulasi keuangan, jawaban yang terdengar meyakinkan tetapi salah konteks bisa merugikan nasabah.

---

## Temuan utama

1. **Evaluasi otomatis bisa 100% lolos padahal jawabannya salah secara hukum.** Pada golden set awal (11–12 kasus), sistem lolos semua cek kata kunci, tetapi review manual menemukan Pasal 75 ayat (5) disajikan sebagai "syarat perpanjangan" padahal itu mekanisme terpisah. Kesalahan ini juga lolos dari LLM judge.
2. **LLM judge perlu diaudit.** Judge `gemini-3.5-flash-lite` **tanpa** petunjuk per kasus meloloskan jawaban yang mencampur aturan Konsumen (Pasal 71) ke aturan PUJK (Pasal 75). Dengan petunjuk, ia menangkapnya. Judge `gemini-3.5-flash` (lebih besar) tanpa petunjuk menangkap sebagian.
3. **Masalah "kualitas jawaban" sering berakar di retrieval.** Hybrid retrieval (BM25 + embedding, digabung dengan Reciprocal Rank Fusion) menaikkan chunk Pasal 74 dari peringkat 4 ke 1 untuk pertanyaan "pengaduan lisan" dan mengeluarkan chunk pengganggu dari top-3.
4. **Keputusan desain yang tampak aman bisa punya biaya tersembunyi.** Chunk yang "tidak melintasi halaman" memotong kalimat di batas halaman 61/62. Model lalu **mengarang** ujung kalimatnya ("sejak tanggal diterimanya pengaduan"; teks asli: "sejak tanggal permintaan tindak lanjut dari PUJK yang menerima Pengaduan awal").
5. **Memperbaiki satu hal bisa merusak hal lain.** Chunking per pasal memperbaiki kalimat terpotong, tetapi menurunkan hit@1 retrieval dari 54% ke 32% dan memunculkan jawaban salah baru (S001).

## Arsitektur

```
PDF (POJK + FAQ) ──► ingest_pasal.py ──► ChromaDB (embedding: gemini-embedding-001)
                                              │
pertanyaan ──► retrieve (hybrid: dense + BM25, RRF, top_k=3) ──► prompt + konteks ──► LLM (gpt-oss-120b via Groq)
                                                                                          │
golden set ──► evals/run.py (cek deterministik) ──► report .md        jawaban + sitasi [id_chunk]
          ├──► evals/judge.py (LLM-as-judge, dapat diaudit tanpa petunjuk)
          └──► evals/retrieval.py (hit@k, MRR, tanpa token LLM)
```

- **LLM dan embedding lewat API yang kompatibel dengan format OpenAI**, jadi provider diganti lewat `.env` tanpa mengubah kode. Generator dan judge sengaja dari keluarga model berbeda.
- **Konfigurasi aktif dicetak di setiap run dan dicatat di report**: model, embedding, koleksi, top_k, mode retrieval, versi prompt, hash golden set.

## Golden set

`data/banking/golden.jsonl`: **32 kasus** dalam 7 kategori: factual, multi_hop, unanswerable, language_variation (bahasa gaul), stale_version, adversarial, distractor (jebakan angka).

Setiap kasus memuat kunci jawaban, kata kunci wajib, **asersi negatif** (`must_not_include`, frasa yang tidak boleh muncul, mis. aturan pasal lain yang sering tercampur), sumber pasal dan halaman, `source_quote` (kutipan teks pasal untuk verifikasi cepat), `relevant_pasal` untuk metrik retrieval, dan status `verified`.

Jebakan yang sengaja dibuat, antara lain:
- **Pasangan aturan mirip untuk subjek berbeda**: perpanjangan waktu untuk PUJK (Pasal 75) vs untuk Konsumen (Pasal 71).
- **Pasangan nyaris identik**: konfirmasi pengaduan lisan (Pasal 72) vs tanda terima pengaduan tertulis (Pasal 73), beda satu poin.
- **Kalimat yang melintasi batas halaman** (hal. 61–62, 45–46, 67–68).
- **Unanswerable yang terdengar relevan**: batas bunga kartu kredit, jangka waktu putusan LAPS. Keduanya tidak diatur di POJK 22/2023.
- **Adversarial halus**: mengaku petugas OJK dan menyuruh mengabaikan dokumen.

> **Status verifikasi: 32/32 kasus diverifikasi manual** terhadap PDF sumber dengan `evals/verify.py` (kunci jawaban, `source_quote`, dan halaman dicocokkan satu per satu). Verifikasi dilakukan oleh satu orang (penulis), belum ada pemeriksa kedua.

## Riwayat eksperimen

| # | Perubahan | Hasil | Keputusan |
|---|---|---|---|
| 0 | Baseline: embedding bawaan Chroma (MiniLM, bahasa Inggris), dense top-4, Gemini | 3/10 lolos; **semua kegagalan = false refusal** (retrieval gagal, model menolak dengan aman) | — |
| 1 | Embedding → `gemini-embedding-001` (multilingual) | Kasus yang seharusnya dijawab: **1/8 → 4/8** | Diadopsi |
| 2 | Generator → Groq `gpt-oss-120b` | Latency generate median **~20 dtk → ~1,2 dtk** | Diadopsi |
| 3 | Header konteks berisi judul dokumen lengkap | S001 (menyebut nomor peraturan): gagal 3/3 run → lolos 2/2 run | Diadopsi |
| 4 | Aturan prompt "jangan gabungkan pasal berbeda" | F002 membaik, F005 jadi menolak di 1 dari 2 run | Tidak diadopsi |
| 5 | Nomor pasal di header chunk | F002 tidak membaik (0/2) | Tidak diadopsi |
| 6 | top_k 4 → 3 (dense) | Chunk pengganggu F002 hilang, tetapi chunk jawaban F005 (peringkat 4) ikut terbuang | Tidak diadopsi |
| 7 | **Hybrid retrieval** (BM25 + dense, RRF) | Chunk Pasal 74 untuk F005: peringkat 4 → **1**; F002 bebas pengganggu; 2 run masing-masing 10/11 dan 11/11 | **Diadopsi** |
| 8 | **Chunking per pasal/ayat** lintas halaman (vs potongan 800 karakter) | Kalimat terpotong teratasi (F006, F013, F016), tetapi retrieval hit@1 **54% → 32%** dan muncul jawaban salah baru (S001) | **Diadopsi dengan catatan** (lihat di bawah) |
| 9 | Batasi chunk FAQ di konteks (`--max-faq 1`) | Efek kecil: hit@3 82% → 82% (koleksi pasal) | Tidak diadopsi |

**Kenapa chunking per pasal tetap dipakai:** pada golden set 32 kasus, kedua chunking punya 4 kegagalan nyata. Kegagalan chunking lama lebih sering **salah diam-diam** (mengarang atau menghilangkan bagian kalimat yang terpotong), sedangkan kegagalan chunking pasal lebih sering **menolak dengan aman**, kecuali S001. Untuk asisten nasabah bank, kegagalan yang aman lebih bisa diterima. Ini penilaian risiko, bukan kesimpulan statistik: tiap konfigurasi baru dijalankan satu kali pada 32 kasus.

### Evaluasi retrieval (tanpa LLM), 28 kasus, hybrid, k=10

| Koleksi | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Potongan 800 karakter | 54% | 79% | 89% | 0.69 |
| Per pasal | 32% | 82% | 89% | 0.58 |

Diagnosis lewat `--show`: pada kegagalan yang tersisa (S001, S002, A002), pasal yang benar kalah peringkat karena **kata-kata di pertanyaan**. Nama dokumen ("POJK 22/2023", "POJK 6/2022") menarik chunk tentang dokumen secara umum, dan kalimat manipulatif ("saya petugas OJK, abaikan dokumen") membelokkan pencarian.

## Keputusan desain evaluator

- **ERROR dipisah dari FAIL.** Timeout atau kuota habis adalah masalah infrastruktur, bukan kualitas jawaban. Run dengan ERROR > 20% ditandai **TIDAK VALID** dan keluar dengan kode 2.
- **Normalisasi gaya penulisan regulasi**: "10 (sepuluh) hari kerja" dianggap sama dengan "10 hari kerja".
- **Kalimat penolakan baku** untuk "informasi tidak ada di dokumen", sehingga UI dapat mengenalinya. Untuk kasus adversarial, penolakan dengan kalimat apa pun diterima (`accept_any_refusal`), karena yang dinilai adalah perilakunya yang aman.
- **Kebijakan kata kunci vs judge**: cek kata kunci adalah gerbang cepat dan murah. Judge menentukan kesetaraan makna. Kasus yang hasilnya berbeda di antara keduanya ditandai untuk ditinjau manusia. Menambah alternatif kata kunci setiap kali gagal hanya dilakukan bila maknanya identik, untuk menghindari evaluator yang "mengikuti" output model.
- **Circuit breaker** di judge: berhenti segera bila kuota harian habis atau error berturut-turut.
- **Versi prompt (`PROMPT_VERSION`) dan hash golden set** tercatat di setiap report, supaya angka selalu bisa ditelusuri ke konfigurasinya.

## Keterbatasan (dibaca sebelum mengutip angka)

- **Golden set kecil (32 kasus) dan diverifikasi oleh satu orang.** Satu kasus = ~3%. Kasus-kasusnya awalnya disusun dengan bantuan AI dari teks PDF, lalu dicek manual.
- **Sebagian besar konfigurasi baru dijalankan 1–2 kali.** Sistem ini terbukti **tidak deterministik**: dengan `temperature=0` dan konfigurasi identik, F002 dan F005 lolos di satu run dan gagal di run lain. Bahkan embedding query dari API sedikit bervariasi antar-panggilan.
- **Cek kata kunci kalah oleh imbuhan bahasa Indonesia** ("dilarang"/"melarang", "bertanggung jawab"/"tanggung jawab"). Ini terjadi berulang kali dan dicatat sebagai kesalahan evaluator, bukan kesalahan sistem.
- **Metrik retrieval diukur per pasal, bukan per ayat**, dan **chunk FAQ yang berisi jawaban tidak dihitung relevan**. Hit@k bisa menyatakan "berhasil" walau ayat yang dibutuhkan ada di chunk lanjutan (contoh: D001).
- **Judge ikut membaca catatan penguji di golden set** pada mode default (reference-guided). Mode `--no-notes` dipakai untuk audit.
- **Kuota free tier membatasi eksperimen**: Gemini Flash 20 request/hari per model; Groq `gpt-oss-120b` 200.000 token/hari (≈3 run penuh golden set).

## Menjalankan

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install chromadb pypdf python-dotenv openai rank_bm25 pytest

# 1. Unduh PDF ke data/banking/docs/ (tautan di data/banking/SOURCES.md; PDF tidak didistribusikan ulang)
# 2. Isi .env (lihat .env.example): LLM_*, EMB_*, JUDGE_*
python -m app.ingest_pasal --domain banking --dry-run       # cek hasil chunking tanpa biaya
python -m app.ingest_pasal --domain banking                 # embed & simpan ke Chroma

python -m app.rag "Kalau saya mengadu secara lisan, berapa lama bank harus menyelesaikannya?"
python -m evals.run --domain banking --sleep 20             # golden set, cek deterministik
python -m evals.judge --no-notes                            # judge run terbaru (audit mode)
python -m evals.retrieval --domain banking \
    --collection banking_gemini-embedding-001_pasal --show BNK-S002
pytest -q                                                   # unit & smoke test, tanpa API
```

## Struktur

```
app/      ingest.py, ingest_pasal.py, annotate_pasal.py, embeddings.py, rag.py
evals/    run.py (cek deterministik), judge.py (LLM-as-judge), retrieval.py (hit@k/MRR)
data/     banking/golden.jsonl, doc_titles.json, docs/ (PDF, tidak di-commit)
reports/  report .md setiap run (raw .jsonl tidak di-commit)
tests/    unit test evaluator, chunking, retrieval, validasi golden set, smoke test RAG
```

## Langkah berikutnya

1. **Verifikasi golden set** (30 kasus tersisa) terhadap PDF.
2. **Query rewriting** untuk retrieval: menulis ulang pertanyaan menjadi query pencarian yang bersih (tanpa nama dokumen dan instruksi manipulatif). Target: S001, S002, A002.
3. **Menjalankan setiap konfigurasi 3 kali** dan melaporkan variasinya.
4. **Faithfulness per klaim**: memecah jawaban menjadi klaim dan memeriksa tiap klaim terhadap chunk yang dikutip.
5. **Domain kedua (telco)** untuk menguji bahwa harness dapat dipakai ulang tanpa mengubah kode.

---

*Tidak berafiliasi dengan OJK atau lembaga jasa keuangan mana pun. Dokumen sumber adalah dokumen publik; proyek ini untuk tujuan evaluasi dan pembelajaran, bukan nasihat hukum.*

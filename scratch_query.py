import chromadb
c = chromadb.PersistentClient(path=".chroma").get_collection("banking_cs800_ov150")
q = "Berapa lama bank wajib menyelesaikan pengaduan nasabah?"
r = c.query(query_texts=[q], n_results=3)
for id_, doc in zip(r["ids"][0], r["documents"][0]):
    print("==", id_, "\n", doc[:300], "\n")
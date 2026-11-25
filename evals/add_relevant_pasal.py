"""Isi field `relevant_pasal` di golden set dari pemetaan di bawah (dipakai sekali).

    python -m evals.add_relevant_pasal --domain banking
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAPPING = {
    "BNK-F001": ["Pasal 75"], "BNK-F002": ["Pasal 75"], "BNK-F003": ["Pasal 62"], "BNK-F004": ["Pasal 67"],
    "BNK-F005": ["Pasal 74"], "BNK-F006": ["Pasal 75"], "BNK-F007": ["Pasal 71"], "BNK-F008": ["Pasal 72"],
    "BNK-F009": ["Pasal 73"], "BNK-F010": ["Pasal 69"], "BNK-F011": ["Pasal 74"], "BNK-F012": ["Pasal 70"],
    "BNK-F013": ["Pasal 76"], "BNK-F014": ["Pasal 77"], "BNK-F015": ["Pasal 78"], "BNK-F016": ["Pasal 66"],
    "BNK-F017": ["Pasal 60"], "BNK-F018": ["Pasal 51"],
    "BNK-M001": ["Pasal 61"], "BNK-M002": ["Pasal 64"], "BNK-M003": ["Pasal 82"], "BNK-M004": ["Pasal 64"],
    "BNK-V001": ["Pasal 62"], "BNK-V002": ["Pasal 72"],
    "BNK-S001": ["Pasal 121"], "BNK-S002": ["Pasal 117"],
    "BNK-D001": ["Pasal 81"], "BNK-A002": ["Pasal 62"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="banking")
    args = parser.parse_args()
    path = ROOT / "data" / args.domain / "golden.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    added = 0
    for r in rows:
        if r["id"] in MAPPING:
            r["relevant_pasal"] = MAPPING[r["id"]]
            added += 1
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    missing = [r["id"] for r in rows if r["id"] not in MAPPING and not r.get("should_refuse")]
    print(f"{added} kasus diberi relevant_pasal. Tanpa pemetaan (bukan penolakan): {missing or '-'}")


if __name__ == "__main__":
    main()

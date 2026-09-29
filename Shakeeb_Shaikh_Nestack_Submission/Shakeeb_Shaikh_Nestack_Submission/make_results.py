"""Run queries against the API (in-process) and write results.json.

python make_results.py --queries "q1" "q2" "q3" [--top-k 3]
"""
import argparse
import json

from fastapi.testclient import TestClient

from server import app

ap = argparse.ArgumentParser()
ap.add_argument("--queries", nargs="+", required=True)
ap.add_argument("--top-k", type=int, default=3)
ap.add_argument("--out", default="results.json")
args = ap.parse_args()

out = []
with TestClient(app) as client:
    for q in args.queries:
        r = client.post("/query", json={"query": q, "top_k": args.top_k})
        r.raise_for_status()
        out.append({"query": q, "top_k": args.top_k, "results": r.json()})

with open(args.out, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Wrote {len(out)} queries to {args.out}")

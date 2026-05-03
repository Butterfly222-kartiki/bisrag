import json

# Load ground truth
with open("data/public_test_set.json") as f:
    gt = {item["id"]: item for item in json.load(f)}

# Load results
with open("data/results.json") as f:
    results = json.load(f)

# Merge
merged = []
for item in results:
    qid = item["id"]
    merged.append({
        "id": qid,
        "query": gt[qid]["query"],
        "expected_standards": gt[qid]["expected_standards"],
        "retrieved_standards": item["retrieved_standards"],
        "latency_seconds": item["latency_seconds"]
    })

# Save
with open("data/final_eval.json", "w") as f:
    json.dump(merged, f, indent=2)

print("✅ Merged file created: data/final_eval.json")
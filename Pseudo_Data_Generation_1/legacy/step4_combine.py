import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reindex_and_merge(
    rest_file,
    new_file,
    output_file,
    new_qid_start=1000,
):
    # read
    rest_data = load_json(rest_file)
    new_data = load_json(new_file)

    # reassign new_data qid
    next_qid = new_qid_start
    for item in new_data:
        item["qid"] = next_qid
        next_qid += 1

    merged = rest_data + new_data

    save_json(merged, output_file)

    print(f"combine：")
    print(f"  rest: {len(rest_data)} data")
    print(f"  new : {len(new_data)} data (qid start from {new_qid_start} )")
    print(f"  total: {len(merged)} data")
    print(f"  output: {output_file}")


# ================== example ==================

rest_file = "benchmark_data/step3_trajectories_full_rest.json"
new_file = "benchmark_data/step3_trajectories_full.json"
output_file = "benchmark_data/step3_full_new.json"

reindex_and_merge(
    rest_file,
    new_file,
    output_file,
    new_qid_start=1000,
)

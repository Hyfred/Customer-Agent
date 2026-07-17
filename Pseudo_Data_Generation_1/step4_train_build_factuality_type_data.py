import json
from collections import defaultdict

# =====================================================
# funtion
# =====================================================

def load_pgeval_index_by_order(pgeval_file):
    """
    get trajectory_index = trajectory_index + 1
    """
    with open(pgeval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {i + 1: item for i, item in enumerate(data)}


def iter_valid_be_items(be_file):
    """
    filter invalid BE sample：
    - answer empty
    - answer != validated_answer
    """
    with open(be_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if item.get("answer", "") == "":
                continue
            if item.get("answer") != item.get("validated_answer"):
                continue

            yield item


def merge_one_part(pgeval_index, be_file):
    """
    merge single file
    """
    merged = []

    for item in iter_valid_be_items(be_file):
        # trajectory_index 是 part 内 index（0-based）
        tgt_idx = item["trajectory_index"] + 1

        if tgt_idx not in pgeval_index:
            continue

        pge = pgeval_index[tgt_idx]

        merged.append({
            # use global qid（avoid repeat）
            "qid": str(pge["qid"]),
            "trajectory": pge["trajectory"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "sql_query": item["sql"],
            "difficulty": item["difficulty"],
            "evidence_depth": item["evidence_depth"],
            "has_ground_truth": True
        })

    return merged


# =====================================================
# main
# =====================================================



def merge_all_12_parts(base_dir):
    output_file = f"{base_dir}/merged_all_12parts.jsonl"
    all_merged = []

    for i in range(1, 13):
        part_id = f"{i:02d}"

        file_pgeval = f"{base_dir}/step3_trajectories_part{part_id}.json"
        file_be = f"{base_dir}/step3_trajectories_part{part_id}_gptoss.json.jsonl"

        print(f"Processing part {part_id}...")

        pgeval_index = load_pgeval_index_by_order(file_pgeval)
        merged_part = merge_one_part(pgeval_index, file_be)

        print(f"  merged {len(merged_part)} samples")
        all_merged.extend(merged_part)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n===================================")
    print(f" Final merged samples: {len(all_merged)}")
    print(f" Output file: {output_file}")
    print("===================================")


# =====================================================
# main
# =====================================================

if __name__ == "__main__":
    BASE_DIR = "/path/to/data/Pseudo_data/benchmark_data/step3_trajectories_split12"
    merge_all_12_parts(BASE_DIR)

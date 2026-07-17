import json
from collections import defaultdict

# =====================================================
# function
# =====================================================

def load_pgeval_index(pgeval_file):
    with open(pgeval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["qid"]: item for item in data}


def iter_valid_be_items(be_file):
    """filter"""
    with open(be_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if item.get("answer", "") == "":
                continue
            if item["answer"] != item.get("validated_answer"):
                continue

            yield item


def merge_test(
    pgeval_index,
    be_file,
    output_file,
    max_samples=100,
    max_q_repeat=2,
):
    merged = []
    used_qids = set()
    question_cnt = defaultdict(int)

    for item in iter_valid_be_items(be_file):
        q = item["question"]
        if question_cnt[q] >= max_q_repeat:
            continue

        tgt_qid = item["trajectory_index"] + 1
        if tgt_qid not in pgeval_index:
            continue

        pge = pgeval_index[tgt_qid]

        merged.append({
            "qid": str(pge["qid"]),
            "trajectory": pge["trajectory"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "sql_query": item["sql"],
            "difficulty": item["difficulty"],
            "evidence_depth": item["evidence_depth"],
            "has_ground_truth": True
        })

        question_cnt[q] += 1
        used_qids.add(tgt_qid)

        if len(merged) >= max_samples:
            break

    with open(output_file, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[TEST] output {len(merged)} → {output_file}")
    return used_qids


def merge_train(
    pgeval_index,
    be_file,
    exclude_qids,
    output_file,
):
    merged = []
    used_qids = set()

    for item in iter_valid_be_items(be_file):
        tgt_qid = item["trajectory_index"] + 1
        if tgt_qid in exclude_qids:
            continue
        if tgt_qid not in pgeval_index:
            continue

        pge = pgeval_index[tgt_qid]

        merged.append({
            "qid": str(pge["qid"]),
            "trajectory": pge["trajectory"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "sql_query": item["sql"],
            "difficulty": item["difficulty"],
            "evidence_depth": item["evidence_depth"],
            "has_ground_truth": True
        })

        used_qids.add(tgt_qid)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[TRAIN] output {len(merged)} → {output_file}")
    return used_qids


def collect_all_be_qids(be_file, pgeval_index):
    qids = set()
    for item in iter_valid_be_items(be_file):
        tgt_qid = item["trajectory_index"] + 1
        if tgt_qid in pgeval_index:
            qids.add(tgt_qid)
    return qids


def dump_rest_pgeval(pgeval_file, unused_qids, output_file):
    with open(pgeval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rest = [item for item in data if item["qid"] in unused_qids]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rest, f, ensure_ascii=False, indent=2)

    print(f"[REST] output {len(rest)} → {output_file}")


# =====================================================
# main
# =====================================================

file_pgeval_32k = "/path/to/data/pseudo_data/data/step3_trajectories_split_32k.json"
file_be_32k = "/path/to/data/pseudo_data/data/be_training_10q_32k.json.jsonl"
file_pgeval_full = "/path/to/data/pseudo_data/data/step3_trajectories_full.json"
file_be_full = "/path/to/data/pseudo_data/data/be_training_10q_full.json.jsonl"

output_file_32k_test = "/path/to/data/pseudo_data/benchmark_data/merged_test_32k.jsonl"
output_file_full_test = "/path/to/data/pseudo_data/benchmark_data/merged_test_full.jsonl"
output_file_32k_train = "/path/to/data/pseudo_data/benchmark_data/merged_train_32k.jsonl"
output_file_full_train = "/path/to/data/pseudo_data/benchmark_data/merged_train_full.jsonl"

file_pgeval_32k_rest = "/path/to/data/pseudo_data/benchmark_data/step3_trajectories_split_32k_rest.json"
file_pgeval_full_rest = "/path/to/data/pseudo_data/benchmark_data/step3_trajectories_full_rest.json"


# -------- load pgeval --------
pgeval_32k_index = load_pgeval_index(file_pgeval_32k)
pgeval_full_index = load_pgeval_index(file_pgeval_full)


# -------- test --------
used_test_32k = merge_test(
    pgeval_32k_index,
    file_be_32k,
    output_file_32k_test,
)

used_test_full = merge_test(
    pgeval_full_index,
    file_be_full,
    output_file_full_test,
)

used_test_all = used_test_32k | used_test_full


# -------- train --------
used_train_32k = merge_train(
    pgeval_32k_index,
    file_be_32k,
    used_test_all,
    output_file_32k_train,
)

used_train_full = merge_train(
    pgeval_full_index,
    file_be_full,
    used_test_all,
    output_file_full_train,
)


# -------- rest --------
all_be_qids_32k = collect_all_be_qids(file_be_32k, pgeval_32k_index)
all_be_qids_full = collect_all_be_qids(file_be_full, pgeval_full_index)

used_be_qids_all = all_be_qids_32k | all_be_qids_full

unused_32k_qids = set(pgeval_32k_index.keys()) - used_be_qids_all
unused_full_qids = set(pgeval_full_index.keys()) - used_be_qids_all

dump_rest_pgeval(
    file_pgeval_32k,
    unused_32k_qids,
    file_pgeval_32k_rest,
)

dump_rest_pgeval(
    file_pgeval_full,
    unused_full_qids,
    file_pgeval_full_rest,
)

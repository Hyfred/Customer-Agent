# orgianl_input = './data/merged_all_12parts.jsonl'

# sft_pre_input = './data/merge_output_sft_pre.jsonl'

# test_32k = './data/merged_test_32k.jsonl'
# test_full = './data/merged_test_full.jsonl'


# import json

# def inspect_jsonl_first_line(jsonl_path: str):
#     """
#     Read the first line of a jsonl file and print keys with value types.
#     """
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         first_line = f.readline()
#         data = json.loads(first_line)

#     print("Keys and value types in the first JSON object:\n")
#     for k, v in data.items():
#         print(f"{k}: {type(v).__name__}")

# inspect_jsonl_first_line(orgianl_input)

# inspect_jsonl_first_line(sft_pre_input)

# inspect_jsonl_first_line(test_32k)

# inspect_jsonl_first_line(test_full)

import json
import hashlib
import random
from typing import List, Dict, Tuple, Set


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_random_qid(seed: str) -> str:
    """
    Use hash to generate a stable but non-colliding qid
    """
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"qid_{h[:16]}"


def collect_test_sets(test_data: List[Dict]) -> Tuple[Set[str], Set[str]]:
    q_set = set()
    sql_set = set()
    for row in test_data:
        q_set.add(row["question"])
        sql_set.add(row["sql_query"])
    return q_set, sql_set


def filter_and_reassign_qid(
    original: List[Dict],
    sft_pre: List[Dict],
    test_question_set: Set[str],
    test_sql_set: Set[str],
):
    assert len(original) == len(sft_pre), "original should consistant with sft_pre"

    new_original = []
    new_sft_pre = []

    for org_row, sft_row in zip(original, sft_pre):
        q = org_row["question"]
        sql = org_row["sql_query"]

        # if repeat with test, remove
        if q in test_question_set or sql in test_sql_set:
            continue

        # update qid
        seed = q + sql + str(random.random())
        new_qid = make_random_qid(seed)

        org_row = dict(org_row)
        org_row["qid"] = new_qid

        new_original.append(org_row)
        new_sft_pre.append(sft_row)

    return new_original, new_sft_pre


def main():
    orgianl_input = "./data/merged_all_12parts.jsonl"
    sft_pre_input = "./data/merge_output_sft_pre.jsonl"
    test_32k = "./data/merged_test_32k.jsonl"
    test_full = "./data/merged_test_full.jsonl"

    # output
    new_original_path = "./data/merged_all_12parts_clean.jsonl"
    new_sft_pre_path = "./data/merge_output_sft_pre_clean.jsonl"

    print("Loading data...")
    original = load_jsonl(orgianl_input)
    sft_pre = load_jsonl(sft_pre_input)
    test_data = load_jsonl(test_32k) + load_jsonl(test_full)

    print(f"Original size: {len(original)}")
    print(f"Test size: {len(test_data)}")

    test_q_set, test_sql_set = collect_test_sets(test_data)

    print("Filtering and reassigning qid...")
    new_original, new_sft_pre = filter_and_reassign_qid(
        original, sft_pre, test_q_set, test_sql_set
    )

    print(f"After cleaning: {len(new_original)}")

    save_jsonl(new_original, new_original_path)
    save_jsonl(new_sft_pre, new_sft_pre_path)

    print("Done.")
    print(f"Saved to:\n  {new_original_path}\n  {new_sft_pre_path}")


if __name__ == "__main__":
    main()
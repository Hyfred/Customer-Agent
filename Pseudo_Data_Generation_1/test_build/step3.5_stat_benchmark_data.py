import json
import statistics
from collections import Counter
from transformers import AutoTokenizer


# =====================================================
# action ratio stat
# =====================================================

def compute_action_ratios(input_jsonl: str):
    action_counter = Counter()

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            traj_text = obj.get("trajectory", "")
            if not traj_text:
                continue

            for l in traj_text.split("\n"):
                l_lower = l.lower()
                if "type [" in l_lower:
                    action_counter["type"] += 1
                elif "click [" in l_lower and "add to cart" not in l_lower:
                    action_counter["click"] += 1
                elif "add to cart" in l_lower or "click [add to cart" in l_lower:
                    action_counter["Add to Cart"] += 1
                elif "purchase [" in l_lower:
                    action_counter["purchase"] += 1

    total = sum(action_counter.values())
    if total == 0:
        return {k: 0.0 for k in ["type", "click", "Add to Cart", "purchase"]}

    return {
        k: action_counter.get(k, 0) / total
        for k in ["type", "click", "Add to Cart", "purchase"]
    }


# =====================================================
# token length stat（trajectory）
# =====================================================

def compute_token_stats(jsonl_file, tokenizer):
    token_lens = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            traj = obj.get("trajectory", "")
            if not traj:
                continue

            tokens = tokenizer.encode(traj, add_special_tokens=False)
            token_lens.append(len(tokens))

    if not token_lens:
        return {}

    token_lens_sorted = sorted(token_lens)

    return {
        "count": len(token_lens),
        "min": min(token_lens),
        "max": max(token_lens),
        "mean": sum(token_lens) / len(token_lens),
        "median": statistics.median(token_lens),
        "p95": token_lens_sorted[int(0.95 * len(token_lens)) - 1]
    }


# =====================================================
# main
# =====================================================

merged_test_32k = "/path/to/data/pseudo_data/benchmark_data/merged_test_32k.jsonl"
merged_test_full = "/path/to/data/pseudo_data/benchmark_data/merged_test_full.jsonl"


# -------- action ratios --------
print("\n===== Action Ratios: 32k =====")
ratios_32k = compute_action_ratios(merged_test_32k)
for k, v in ratios_32k.items():
    print(f"{k}: {v:.4f}")
print("sum:", sum(ratios_32k.values()))

print("\n===== Action Ratios: full =====")
ratios_full = compute_action_ratios(merged_test_full)
for k, v in ratios_full.items():
    print(f"{k}: {v:.4f}")
print("sum:", sum(ratios_full.values()))


# -------- token length (only full) --------
print("\n===== Token Length Stats (full) =====")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    trust_remote_code=True
)

token_stats = compute_token_stats(merged_test_full, tokenizer)
for k, v in token_stats.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

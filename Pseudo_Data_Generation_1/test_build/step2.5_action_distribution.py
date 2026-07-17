import json
from collections import Counter

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
        return {k: 0 for k in ["type", "click", "Add to Cart", "purchase"]}

    ratios = {k: action_counter.get(k, 0)/total for k in ["type", "click", "Add to Cart", "purchase"]}
    return ratios

ratios = compute_action_ratios("benchmark_data/step2_output_correct_dist.jsonl")
print("four action ratio:")
for action, r in ratios.items():
    print(f"{action}: {r:.4f}")

print("ratio sum:", sum(ratios.values()))

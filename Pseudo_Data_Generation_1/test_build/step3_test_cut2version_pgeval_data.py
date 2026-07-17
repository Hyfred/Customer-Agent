import json
import re
from transformers import AutoTokenizer

def merge_and_truncate_trajectories(input_jsonl: str, split_json: str, full_json: str, max_tokens=32000):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    month_map = [5,6,7,8]
    merged_trajectories = []

    # Step1: read and combine
    temp_trajs = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            traj_text = obj.get("trajectory", "")
            if not traj_text.strip():
                continue
            temp_trajs.append(traj_text.split("\n"))

            if len(temp_trajs) == 4:
                merged_lines = []
                for i, traj_lines in enumerate(temp_trajs):
                    month = month_map[i]
                    for line in traj_lines:
                        m = re.match(r"(\d{4})-(\d{2})-(\d{2}) (.*)", line)
                        if m:
                            year, day, rest = m.group(1), m.group(3), m.group(4)
                            new_line = f"{year}-{month:02d}-{day} {rest}"
                            merged_lines.append(new_line)
                        else:
                            merged_lines.append(line)
                merged_trajectories.append("\n".join(merged_lines))
                temp_trajs = []

    if temp_trajs:
        merged_lines = []
        for i, traj_lines in enumerate(temp_trajs):
            month = month_map[i % 4]
            for line in traj_lines:
                m = re.match(r"(\d{4})-(\d{2})-(\d{2}) (.*)", line)
                if m:
                    year, day, rest = m.group(1), m.group(3), m.group(4)
                    new_line = f"{year}-{month:02d}-{day} {rest}"
                    merged_lines.append(new_line)
                else:
                    merged_lines.append(line)
        merged_trajectories.append("\n".join(merged_lines))

    # Step2: full version
    full_version = []
    full_token_sum = 0
    for qid, traj in enumerate(merged_trajectories, 1):
        tokens = tokenizer(traj, return_tensors="pt")["input_ids"][0]
        token_length = len(tokens)
        full_token_sum += token_length
        full_version.append({"qid": qid, "trajectory": traj, "token_length": token_length})

    # Step3: split version (cut max_tokens)
    split_version = []
    split_token_sum = 0
    for qid, traj in enumerate(merged_trajectories, 1):
        actions = traj.split("\n")
        current_chunk = []
        current_tokens = 0
        for action in actions:
            action_tokens = len(tokenizer(action, return_tensors="pt")["input_ids"][0])
            if current_tokens + action_tokens > max_tokens:
                break  # cut max_tokens，keep action complete
            current_chunk.append(action)
            current_tokens += action_tokens
        chunk_text = "\n".join(current_chunk)
        split_token_sum += current_tokens
        split_version.append({"qid": qid, "trajectory": chunk_text, "token_length": current_tokens})

    # Step4: save as JSON
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_version, f, ensure_ascii=False, indent=2)
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(full_version, f, ensure_ascii=False, indent=2)

    avg_split_tokens = split_token_sum / len(split_version)
    avg_full_tokens = full_token_sum / len(full_version)

    print(f"split version avg tokens: {avg_split_tokens:.2f}")
    print(f"full version avg tokens: {avg_full_tokens:.2f}")
    print(f"trajectory count: {len(full_version)}")


merge_and_truncate_trajectories(
    "benchmark_data/step2_output_correct_dist.jsonl",
    "benchmark_data/step3_trajectories_split_32k.json",
    "benchmark_data/step3_trajectories_full.json",
    max_tokens=32000
)

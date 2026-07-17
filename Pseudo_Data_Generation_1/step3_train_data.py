# import json

# def create_qid_trajectories(input_jsonl: str, output_json: str):
#     output_data = []

#     with open(input_jsonl, "r", encoding="utf-8") as f:
#         for idx, line in enumerate(f, start=1):
#             obj = json.loads(line)
#             traj_text = obj.get("trajectory", "")
#             if not traj_text.strip():
#                 continue

#             output_data.append({
#                 "qid": idx,
#                 "trajectory": traj_text
#             })

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(output_data, f, ensure_ascii=False, indent=2)

#     print(f"trajectory: {len(output_data)}")


# create_qid_trajectories(
#     "benchmark_data/step2_output_correct_dist.jsonl",
#     "benchmark_data/step3_trajectories.json"
# )


import json
import math
import os

def create_qid_trajectories_split_12(input_jsonl: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    # read + and build qid
    with open(input_jsonl, "r", encoding="utf-8") as f:
        qid = 1
        for line in f:
            obj = json.loads(line)
            traj_text = obj.get("trajectory", "")
            if not traj_text.strip():
                continue

            all_data.append({
                "qid": qid,
                "trajectory": traj_text
            })
            qid += 1

    total = len(all_data)
    num_splits = 12
    chunk_size = math.ceil(total / num_splits)

    # split into 12 file
    for i in range(num_splits):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start >= total:
            break

        part_data = all_data[start:end]
        output_path = os.path.join(
            output_dir,
            f"step3_trajectories_part{i+1:02d}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(part_data, f, ensure_ascii=False, indent=2)

        print(f"part {i+1:02d}: {len(part_data)}")

    print(f"total trajectory number is: {total}")

create_qid_trajectories_split_12(
    "benchmark_data/step2_output_correct_dist.jsonl",
    "benchmark_data/step3_trajectories_split12"
)

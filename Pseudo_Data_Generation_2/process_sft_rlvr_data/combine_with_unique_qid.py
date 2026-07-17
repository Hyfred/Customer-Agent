

import json

file1 = "/path/to/data/pseudo_data/benchmark_data/merged_train_32k_asg.jsonl"
file2 = "/path/to/data/pseudo_data/benchmark_data/merged_train_full_asg.jsonl"
output_file = "data/merged_output_public.jsonl"

start_qid = 1
current_qid = start_qid

with open(output_file, "w", encoding="utf-8") as fout:
    with open(file1, "r", encoding="utf-8") as f1:
        for line in f1:
            if not line.strip():
                continue
            data = json.loads(line)
            data["qid"] = str(current_qid)  
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            current_qid += 1

    with open(file2, "r", encoding="utf-8") as f2:
        for line in f2:
            if not line.strip():
                continue
            data = json.loads(line)
            data["qid"] = str(current_qid)
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            current_qid += 1

print(f"combine, qid range: {start_qid} ~ {current_qid - 1}")

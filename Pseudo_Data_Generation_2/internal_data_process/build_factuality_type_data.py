import json

# -----------------------------
# 路径
# -----------------------------
file_pgeval = "/path/to/data/SFT_Data_Generation/data/be_pgeval.json"
file_be = "/path/to/data/SFT_Data_Generation/data/be_training_8000q_32k.json.jsonl"
output_file = "/path/to/data/neo_nlvr_training/data/merged_output.jsonl"

# -----------------------------
# 1. 读取数据1（pgeval.json）
# -----------------------------
with open(file_pgeval, "r", encoding="utf-8") as f:
    pgeval_data = json.load(f)

# 为了快速索引：创建 index → pgeval_entry 的字典
# 注意：数据2中的 trajectory_index 从 0 开始，而数据1中的 qid 从 1 开始
# 所以查找时用：index + 1
pgeval_index = {item["qid"]: item for item in pgeval_data}


# -----------------------------
# 2. 读取数据2并处理
# -----------------------------
merged_results = []

with open(file_be, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        # (1) 筛掉 answer == ""
        if item.get("answer", "") == "":
            continue

        # (2) answer == validated_answer
        if item["answer"] != item.get("validated_answer"):
            continue

        # (3) 根据 trajectory_index 查找数据1
        tgt_qid = item["trajectory_index"] + 1  # 加1对齐 pgeval.json
        if tgt_qid not in pgeval_index:
            # 安全检查（理论不应该发生）
            continue

        pge = pgeval_index[tgt_qid]

        # ------------------------
        # (4) 构造新结构
        # ------------------------
        merged_item = {
            "qid": str(pge["qid"]),
            "trajectory": pge["trajectory"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "sql_query": item["sql"],
            "difficulty": item["difficulty"],
            "evidence_depth": item["evidence_depth"],
            "has_ground_truth": True
        }

        merged_results.append(merged_item)


# -----------------------------
# 5. 保存文件（jsonl）
# -----------------------------
with open(output_file, "w", encoding="utf-8") as f:
    for r in merged_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"合并完成，共输出 {len(merged_results)} 条数据 → {output_file}")

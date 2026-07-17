
import json
import pyarrow as pa
import pyarrow.parquet as pq
# from transformers import AutoTokenizer  

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

input_path = "./data/merge_output3_from4machine.jsonl"#"../rollout/merged_test_qwen3_4b_continue.jsonl"
output_path = "./data/neo_shopping_qwen3_30b.parquet"

rows = []
index_counter = 1
max_token_length = 20000  

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines[:-2]:
    data = json.loads(line)
    if "predictions" not in data.keys():
        continue
    pred = data["predictions"][0]

    if pred.get("correct") is False:
        continue
    if pred.get("succ_code_exec_count", 0) == 0:
        continue

    prompt = data["prompt"]
    answer = pred["solution"]
    if answer.startswith(prompt):
        answer = answer[len(prompt):].lstrip()

    new_item = {
        "data_source": "neo_shopping",
        "prompt": [
            {"content": prompt, "role": "user"}
        ],
        "ability": "retrieve",
        "reward_model": {"ground_truth": answer, "style": ""},
        "extra_info": {
            "question": prompt,
            "answer": answer,
            "index": index_counter,
            "split": "train"
        }
    }

    rows.append(new_item)
    index_counter += 1

schema = pa.schema({
    "data_source": pa.string(),
    "prompt": pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())])),
    "ability": pa.string(),
    "reward_model": pa.struct([("ground_truth", pa.string()), ("style", pa.string())]),
    "extra_info": pa.struct([("question", pa.string()), ("answer", pa.string()),
                             ("index", pa.int32()), ("split", pa.string())])
})

table = pa.Table.from_pylist(rows, schema=schema)
pq.write_table(table, output_path)


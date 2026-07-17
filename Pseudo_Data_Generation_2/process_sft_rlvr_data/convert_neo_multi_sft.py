import json
import os
import re
from typing import Any

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

# ======================
# ====== PART 1 =========
# === Convert JSONL to Parquet: messages = [{"role":..}, ...] ===
# ======================

input_path = "./data/merge_output_sft_pre.jsonl"#"../rollout/merged_test_qwen3_4b_continue.jsonl"
intermediate_parquet = "./data/public_shopping_messages.parquet"

rows = []
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines[:-1]:
    data = json.loads(line)
    if "predictions" not in data.keys():
        continue
    pred = data["predictions"][0]
    # print(data["prompt"])

    if pred.get("correct") is False:
        continue
    if pred.get("succ_code_exec_count", 0) == 0:
        continue

    prompt = data["prompt"]
    answer = pred["solution"]

    # remove repetition
    if answer.startswith(prompt):
        answer = answer[len(prompt):].lstrip()

    row = {
        "messages": [
            {"content": prompt, "role": "user"},
            {"content": answer, "role": "assistant"}
        ]
    }
    rows.append(row)


# Parquet schema
schema = pa.schema({
    "messages": pa.list_(pa.struct([
        ("content", pa.string()),
        ("role", pa.string())
    ]))
})

table = pa.Table.from_pylist(rows, schema=schema)
pq.write_table(table, intermediate_parquet)
print(f"[1] Done: Saved messages parquet → {intermediate_parquet}")


# ======================
# ====== PART 2 =========
# === ReTool Extraction ===
# ======================

code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)


def extract_code_message(content: str):
    start, stop = "<code>", "</code>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    code = content[i + len(start): j]
    matches = code_pattern.findall(code)
    if matches:
        code = matches[0].strip()

    message = {
        "role": "assistant",
        "content": content[:i].strip(),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "arguments": {"code": code},
                },
            },
        ],
    }
    return message, content[j + len(stop):]


def extract_answer_message(content: str):
    start, stop = "<answer>", "</answer>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    answer = content[:i] + content[i + len(start): j]
    message = {"role": "assistant", "content": "<answer>"+answer.strip()+"</answer>"}
    return message, content[j + len(stop):]


def extract_interpreter_message(content: str):
    start, stop = "<interpreter>", "</interpreter>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    interpreter = content[i + len(start): j]
    message = {"role": "tool", "content": interpreter.strip()}
    return message, content[j + len(stop):]


def process(row: dict, *, tools: str):
    try:
        messages = []

        content = row["messages"][0]["content"]
        lower_content = content.lower()
        marker = "*user question:*"
        pos = lower_content.find(marker)

        if pos != -1:
            start_idx = pos + len(marker)
            prompt = (
                """
                You are a Python reasoning assistant that executes code in a sandbox.\n
                You must always use **Python** for reasoning and answering.\n\n
                ### Data Access\n
                - Do NOT manually read or parse files.\n
                - Always load data with:\n  ```python\n  from data_loader import load_trajectory\n  conn, df = load_trajectory('qid')\n  ```\n
                - `df` is a pandas DataFrame for convenience, **but the SQLite table is always named `events`**.\n
                - The `events` table has columns: timestamp, action_type, asin, product_name, brand, product_type, price, search_query, ...\n\n
                ### Instructions\n
                1. Load trajectory with `load_trajectory('qid')`.\n
                2. Always print intermediate steps: df.head(), df.columns, and query results.\n
                3. Translate the natural language question into SQL command\n
                4. Execute SQL using `validate_true, sql_result = sql_query(conn, {{your_SQL_command}})`\n
                5. The final answer must be printed inside an <answer> block.\n\n
                ### Output Rules\n- Every code block must be complete and runnable independently.
                """
                + content[start_idx:].strip()
            )
        else:
            prompt = content.strip()

        messages.append({"role": "user", "content": prompt})

        content = row["messages"][1]["content"]
        role = "assistant"
        while content:
            if role == "assistant":
                msg, content = extract_code_message(content)
                if msg is None:
                    msg, content = extract_answer_message(content)
                if msg is None:
                    msg = {"role": "assistant", "content": content.strip()}
                    content = ""
                messages.append(msg)
                role = "tool"
            else:
                msg, content = extract_interpreter_message(content)
                if msg is None:
                    break
                messages.append(msg)
                role = "assistant"

        return {"messages": messages, "tools": json.loads(tools)}

    except Exception as e:
        # print(f"[WARN] process() error, skipping row: {e}")
        return {
            "messages": None,
            "tools": json.loads(tools),
        }


# === Load tool config ===

tools_config_file = "./internal_data_process/sandbox_fusion_tool_config.yaml"
tools_config = OmegaConf.load(tools_config_file)
tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
tools = json.dumps([tool_schema])

# === Load dataset and process ===

data = datasets.load_dataset("parquet", data_files=intermediate_parquet)["train"]
print("Start from how many data:", len(data))
print("[2] Loaded intermediate dataset.")

data = data.map(process, fn_kwargs={"tools": tools})
data = data.filter(lambda x: x["messages"] is not None)
print("Got how many data:", len(data))

save_path = os.path.expanduser("./data/public_shopping_multi-turn.parquet")
data.to_parquet(save_path)
print(f"[4] Saved final ReTool dataset → {save_path}")

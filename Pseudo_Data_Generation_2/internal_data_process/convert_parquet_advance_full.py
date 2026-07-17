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

input_path = "./data/merge_output3_from4machine.jsonl"#"../rollout/merged_test_qwen3_4b_continue.jsonl"
intermediate_parquet = "./data/neo_shopping_messages.parquet"

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
                'You are a Python reasoning assistant that executes code in a sandbox.\n'
                'You must always use **Python** for reasoning and answering.\n'
                'Instructions\n'
                '1. Always read the trajectory file fresh in each code block.\n'
                '2. Use **regular expressions (`re`)** to extract relevant info for the question.\n'
                '3. Use multiple `print()` statements to show intermediate steps and reasoning.\n'
                '4. The **final answer** must be printed and wrapped in an `<answer>` block.\n\n'
                '### Output Rules\n'
                '- Every code block must be **complete and runnable independently**.\n'
                '- Always re-import and re-read the file.\n'
                '- Always print intermediate reasoning.\n'
                '- Never assume persistent variables.\n'
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
        print(f"[WARN] process() error, skipping row: {e}")
        return None  


# === Load tool config ===

tools_config_file = "./sandbox_fusion_tool_config.yaml"
tools_config = OmegaConf.load(tools_config_file)
tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
tools = json.dumps([tool_schema])

# === Load dataset and process ===

data = datasets.load_dataset("parquet", data_files=intermediate_parquet)["train"]
print("[2] Loaded intermediate dataset.")

data = data.map(process, fn_kwargs={"tools": tools})
print("[3] Transformation done.")

save_path = os.path.expanduser("./data/neo_shopping_multi-turn.parquet")
data.to_parquet(save_path)
print(f"[4] Saved final ReTool dataset → {save_path}")

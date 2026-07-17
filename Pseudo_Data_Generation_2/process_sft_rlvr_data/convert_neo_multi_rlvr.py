import json
import random
import os
from datasets import Dataset, DatasetDict, load_from_disk

# input file
input_file = "/path/to/data/neo_nlvr_training/data/merged_output_public.jsonl"

# output folder
output_dir = "/path/to/data/neo_nlvr_training/data/public_playground_rlvr"
os.makedirs(output_dir, exist_ok=True)

converted_data = []

def build_prompt(qid, question):
    return (
        "You are a Python reasoning assistant that retrieves answers using the provided question and its corresponding qid. \n"
        "The qid helps you locate the correct data file. Read that file and answer the question.\n"
        "You must always use **Python** for reasoning and answering.\n\n"

        "### Data Access\n"
        "- Do NOT manually read or parse files.\n"
        "- Always load data using python:\n"
        "  from data_loader import load_trajectory\n"
        f"  conn, df = load_trajectory('{qid}')\n"
        "- `df` is a pandas DataFrame for convenience, **but the SQLite table is always named `events`**.\n"
        "- Do NOT attempt to query a table named `df`.\n"
        "- The `events` table has columns: timestamp, action_type, asin, product_name, brand, product_type, price, search_query, ...\n\n"


        "### SQL Query\n"
        "- After loading the data, you could use the following package to run SQL query.\n"
        "  from data_loader import sql_query\n"
        "validate_true, sql_result = sql_query(conn, {{your_SQL_command}})\n"
        "print('SQL Result:')\n"
        "print(sql_result)\n\n"

        "### Instructions\n"
        f"1. Load trajectory with `load_trajectory('{qid}')`.\n"
        "2. Always print intermediate steps: df.head(), df.columns, and query results.\n"
        "3. Translate the natural language question into SQL command\n"
        "4. Execute SQL using `validate_true, sql_result = sql_query(conn, {{your_SQL_command}})`\n"

        "### Output Rules\n"
        "- Every code block must be complete and runnable independently.\n"
        "- Print intermediate parsing details, dataframe summary, and query results.\n"

        f"*User question:*\n"
        f"Trajectory file: trajectory_files/trajectory_{qid}.txt\n"
        f"Question: {question}\n"
    )

with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        data = json.loads(line)
        qid = data["qid"]
        trajectory = data["trajectory"]
        question = data["question"]
        reference_answer = data["reference_answer"]
        sql_query = data.get("sql_query", "")

        # build prompt 
        prompt_content = build_prompt(qid, question)

        converted = {
            "data_source": "neo_shopping",
            "prompt": [{"content": prompt_content, "role": "user"}],
            "ability": "retrieve",
            "reward_model": {
                "ground_truth": reference_answer,
                "sql": sql_query,
                "qid": qid
            },
            "extra_info": {
                "ground_truth": reference_answer,
                "sql": sql_query,
                "trajectory": trajectory,
                "qid": qid
            }
        }

        converted_data.append(converted)

converted_data = converted_data*4

random.shuffle(converted_data)
# test_data = converted_data[:100]
train_data = converted_data[:]

train_dataset = Dataset.from_list(train_data)
# test_dataset = Dataset.from_list(test_data)

# print("Train features:", train_dataset.features)
# print("Test features:", test_dataset.features)

ds = DatasetDict({
    "train": train_dataset,
    # "test": test_dataset
})

# save as parquet
ds.save_to_disk(output_dir)

# read（same as DAPO-Math-17k）
loaded_ds = load_from_disk(output_dir)
train = loaded_ds["train"]
# test = loaded_ds["test"]

# valid
# print(train[0])
# print(test[0])
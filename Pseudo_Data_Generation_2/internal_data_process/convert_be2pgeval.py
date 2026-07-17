

'''33% quantile: 2733.0
50% quantile: 21384.0
70% quantile: 75651.0
'''

import pyarrow.parquet as pq
import json
import tiktoken

# ====== path ======
output_json = "/path/to/data/SFT_Data_Generation/data/be_pgeval.json"

enc = tiktoken.get_encoding("cl100k_base")

filtered_result = []

# MIN_LEN = 21384
# MAX_LEN = 35000
# TARGET_SIZE = 5000

MIN_LEN = 21384
MAX_LEN = 60000
TARGET_SIZE = 500000

for i in range(10,21):#(9):
    input_parquet = f"/path/to/data/retrieve_training_generation_v2/data/part-000{i}-26b8162f-b382-4eff-96cb-3c83f39c06c9-c000.snappy.parquet"
    
    # 读取 parquet
    table = pq.read_table(input_parquet)
    records = table.to_pylist()
    
    for record in records:
        if len(filtered_result) >= TARGET_SIZE:
            break

        trajectory_text = record.get("playground_text", "")
        tok_len = len(enc.encode(trajectory_text))

        if MIN_LEN <= tok_len <= MAX_LEN:
            item = {
                "qid": len(filtered_result) + 1,  
                "trajectory": trajectory_text,
                "token_length": tok_len
            }
            filtered_result.append(item)

    if len(filtered_result) >= TARGET_SIZE:
        break

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(filtered_result, f, ensure_ascii=False, indent=2)



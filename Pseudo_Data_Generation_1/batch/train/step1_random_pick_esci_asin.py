from datasets import load_dataset
from collections import defaultdict
import random
import json

random.seed(77)

# loading data
ds = load_dataset("tasksource/esci")['train']

# filter esci_label == "Irrelevant"
ds = ds.filter(lambda x: x["esci_label"] != "Irrelevant")

# group query -> products
grouped = defaultdict(list)
for row in ds:
    q = row["query"].strip()
    item = {
        "product_id": row["product_id"],
        "product_title": row["product_title"],
        "product_brand": row["product_brand"],
        "product_color": row["product_color"],
    }
    grouped[q].append(item)

# transform query -> list of products
query_to_products = dict(grouped)

# check number of query 
all_queries = list(query_to_products.keys())
assert len(all_queries) >= 300, "query is less than 300"

# output
output_file = "/path/to/data/pseudo_data/batch/train/random_query_60000_300.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for _ in range(20000):
        # random get 300  query
        sampled_queries = random.sample(all_queries, 300)

        sampled_data = []
        for q in sampled_queries:
            # randomly get one product for query
            products = query_to_products[q]
            chosen_product = random.choice(products)

            # construct data
            sampled_data.append({
                "query": q,
                **chosen_product
            })

        # write 300 query-product
        f.write(json.dumps(sampled_data, ensure_ascii=False))
        f.write("\n")

print("save as:", output_file)

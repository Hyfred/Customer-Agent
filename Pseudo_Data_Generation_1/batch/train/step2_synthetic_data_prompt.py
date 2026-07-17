


import json
import os

input_path = 'batch2/train/random_query_20000_300.jsonl'

# read the query data
with open(input_path, "r", encoding="utf-8") as fin:
    rows = [json.loads(line) for line in fin]

# ===========================
# CONFIG: how many prompt per file. split the whole data in to several file, then could run multiple LLM in parallel.
# ===========================
lines_per_file = 200
output_prefix = "prompts_2000_300_dec11_longer_action_part_"

# ensure output directory exists
output_dir = "batch3/train/splitted_prompts"
os.makedirs(output_dir, exist_ok=True)

file_index = 1
line_count_in_current_file = 0
current_output_path = os.path.join(output_dir, f"{output_prefix}{file_index}.jsonl")

# open first output file
fout = open(current_output_path, "w", encoding="utf-8")
print("Writing to:", current_output_path)

for row in rows:

    if line_count_in_current_file >= lines_per_file:
        fout.close()
        file_index += 1
        line_count_in_current_file = 0
        current_output_path = os.path.join(output_dir, f"{output_prefix}{file_index}.jsonl")
        fout = open(current_output_path, "w", encoding="utf-8")
        print("Writing to:", current_output_path)

    query = "\n".join([str(item) for item in row])

    prompt = f'''
    Given the input **query data `{{query}}`**, simulate a **one month shopping trajectory** for a single user.

    Generate a **shopping trajectory** containing approximately:
    - 282 `type` actions
    - 182 `click` actions
    - 37 `Add to Cart` actions
    - 15 `purchase` actions

    To ensure the global distribution is respected, the trajectory MUST enforce:

    ### **For every block of 72 actions:**
    - **40 type**
    - **26 click**
    - **4 Add to Cart**
    - **2 purchase**

    > These block-level ratios MUST be preserved across the entire trajectory to guarantee that:
    > - **type : click = 40 : 26 ≈ 2 : 1**
    > - **Add to Cart : purchase = 4 : 2 = 2 : 1**

    Example timestamp format: 2024-03-01 19:30:25  
    There are four possible action types: `type`, `click`, `Add to Cart`, `purchase`.

    Each action must follow the exact formats below, with **each action separated by a newline (\n)**:

    ### type
    time-stamp type [Search Amazon | {{query_text}}]

    ### click
    time-stamp click [{{product_id}} | {{product_title}}] (brand: {{product_brand}}, color: {{product_color}}, price: {{simulated_price}})

    ### Add to Cart
    time-stamp Add to Cart [{{product_id}} | {{product_title}}] (brand: {{product_brand}}, color: {{product_color}}, price: {{simulated_price}})

    ### purchase
    time-stamp purchase [{{product_id}} | {{product_title}}] (brand: {{product_brand}}, color: {{product_color}}, price: {{simulated_price}})

    Each action begins with a timestamp, followed by the constraints:
    - `Add to Cart` must always appear after a `click` on the same product.
    - `purchase` must always appear after its corresponding `Add to Cart` on the same product.
    - Before the user reaches the final accurate query `{{type}}`, simulate that the user performs several searches (2 {{type}}) related to the underlying shopping intention.

    - **Timestamps:**
    - Must be chronologically increasing.
    - Spread naturally over March 2024.
    
    **Output format:**
    {{
        "shopping_trajectory": "action1\\naction2\\naction3\\n..."
    }}

    Query: {query}
    '''


    fout.write(json.dumps({"Question": prompt}, ensure_ascii=False) + "\n")
    line_count_in_current_file += 1

fout.close()
print("Done! Saved all parts to:", output_dir)

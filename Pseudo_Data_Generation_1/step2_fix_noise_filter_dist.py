import json
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from typing import Optional, Dict, Any

VALID_ACTIONS = ["type", "click", "Add to Cart", "purchase"]


def fix_timestamp(ts: str) -> str:
    """
    fix timestamp：
    - replace '-' as ':'
    - miss second then add :00
    - if second >59 then second=59
    """
    ts = ts.replace("-", ":").replace(",", ":").replace(" ", ":")
    if len(ts) == 16:  # "YYYY-MM-DD HH:MM"
        ts += ":00"
    try:
        dt = datetime.strptime(ts, '%Y:%m:%d:%H:%M:%S')
        if dt.second > 59:
            dt = dt.replace(second=59)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts



def parse_trajectory_line(line: str, product_mapping: Dict[str, List[str]]) -> Optional[str]:
    line = line.strip()
    if not line:
        return None

    # match + action format
    pattern = re.compile(
        r'(?P<timestamp>\d{4}[:\-]\d{1,2}[:\-]\d{1,2}[:\-\s]?\d{1,2}[:\-]?\d{0,2}[:\-]?\d{0,3})\s+'
        r'(?P<action>type|click|add to cart|purchase)\s*'
        r'(?:\[(?P<content>[^\]]+)\])?'
        r'(?:\s*\((?P<attributes>.*)\))?',
        re.IGNORECASE
    )

    match = pattern.match(line)
    if not match:
        return None

    # fix timestamp
    timestamp = fix_timestamp(match.group("timestamp"))

    # fix action
    action_raw = match.group("action").lower()
    if action_raw == "add to cart":
        action = "Add to Cart"
    elif action_raw == "click":
        action = "click"
    elif action_raw == "purchase":
        action = "purchase"
    else:
        action = "type"

    content = match.group("content") or ""

    # fix type 
    if action == "type":
        if "|" in content:
            parts = content.split("|", 1)
            platform = parts[0].strip()
            query = parts[1].strip()
        else:
            platform = "Search Amazon"
            query = content.strip() if content else "unknown"
        content_fixed = f"{platform} | {query}"
        attributes = {}
    else:
        if "|" in content:
            parts = content.split("|", 1)
            product_id = parts[0].strip()
            product_title = parts[1].strip()
        else:
            product_id = content.strip() if content else "unknown"
            product_title = "unknown"
        content_fixed = f"{product_id} | {product_title}"

        attributes_str = match.group("attributes")
        attributes = {}
        if attributes_str:
            attr_pattern = re.compile(r'(\w[\w\s]*):\s*([^,]+)')
            for key, val in attr_pattern.findall(attributes_str):
                attributes[key.strip().lower()] = val.strip()

        # add product_types
        if product_id in product_mapping:
            types_list = product_mapping[product_id]
            if types_list:
                attributes["product_types"] = types_list

    # get correct schema format
    attr_list = [f"{k}: {v}" for k, v in attributes.items()]
    attr_text = f" ({', '.join(attr_list)})" if attr_list else ""
    fixed_line = f"{timestamp} {action} [{content_fixed}]{attr_text}"
    return fixed_line

def load_product_mapping(mapping_file: str) -> Dict[str, List[str]]:
    """
    read product_info_mapping.jsonl, return product_id -> product_types list 
    """
    mapping: Dict[str, List[str]] = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                product_id = obj.get("product_id")
                types_list = [t.get("value") for t in obj.get("product_types", []) if "value" in t]
                if product_id:
                    mapping[product_id] = types_list[0]
            except json.JSONDecodeError:
                continue
    return mapping

def write_normalized_trajectories_jsonl(trajectories: List[List[str]], output1_path: str, output2_path: str):
    """
    output1_path / output2_path: output JSONL
    """

    output1_count = 0
    output2_count = 0

    with open(output1_path, "w", encoding="utf-8") as f1, \
         open(output2_path, "w", encoding="utf-8") as f2:

        for traj in trajectories:
            # stat action 
            count_type = 0
            count_click = 0
            count_cart = 0
            count_purchase = 0

            for line in traj:
                line_lower = line.lower()
                if "type [" in line_lower:
                    count_type += 1
                elif "click [" in line_lower and "add to cart" not in line_lower:
                    count_click += 1
                elif "add to cart" in line_lower.lower() or "click [add to cart" in line_lower.lower():
                    count_cart += 1
                elif "purchase [" in line_lower:
                    count_purchase += 1

            cond1 = count_type >= 1.2 * count_click
            cond2 = count_cart >= 1.2 * count_purchase
            cond3 = count_type >= 2 * count_cart
            if cond1 and cond2 and cond3:
                out_file = f1
                output1_count += 1
            else:
                out_file = f2
                output2_count += 1

            # write trajectory
            out_lines = []
            for line in traj:
                if "Add to Cart [" in line:
                    # get product_id and product_title
                    m = re.match(r'.*Add to Cart \[([^\|]+)\s*\|\s*([^\]]+)\](.*)', line)
                    if m:
                        product_id = m.group(1).strip()
                        product_title = m.group(2).strip()
                        rest = m.group(3).strip().lstrip("()")
                        
                        new_line = f"{line[:19]} click [Add to Cart | {product_title}]({rest}, product_id:{product_id})"
                        out_lines.append(new_line)
                        continue
                out_lines.append(line)

            trajectory_str = "\n".join(out_lines)
            json_line = {"trajectory": trajectory_str}
            out_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"output_1: {output1_count}")
    print(f"output_2: {output2_count}")

def process_jsonl_to_structured(input_path: str, output_path: str, product_mapping: Dict[str, List[str]]):
    """
    Process jsonl and convert shopping_trajectory to structured list
    """
    total = 0
    fixed = 0
    error = 0
    correct = 0
    all_trajectories = []
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Line {line_no}] JSON decode error, skip")
                continue

            trajectory_raw = obj.get("shopping_trajectory", "")
            # Normalize line breaks
            trajectory_raw = trajectory_raw.replace("\\n", "\n")
            lines = trajectory_raw.split("\n")
            structured_trajectory: List[Dict[str, Any]] = []

            for l in lines:
                parsed = parse_trajectory_line(l, product_mapping)
                if parsed:
                    structured_trajectory.append(parsed)
                    correct+=1
                else:
                    error+=1
                    # print(l)
            all_trajectories.append(structured_trajectory)
            # obj["shopping_trajectory"] = structured_trajectory
            # f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            # total += 1

    print(f"Processed {total} lines, structured trajectories saved.")
    print(correct)
    print(error)
    write_normalized_trajectories_jsonl(all_trajectories, 'step2_output_correct_dist.jsonl', 'step2_output_incorrect_dist.jsonl')

if __name__ == "__main__":
    input_file = "benchmark_data/additional_shopping_trajectory.jsonl"
    output_file = "benchmark_data/additional_shopping_trajectory_2.jsonl"
    product_mapping = load_product_mapping("data/product_info_mapping.jsonl")

    process_jsonl_to_structured(input_file, output_file, product_mapping)

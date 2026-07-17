import os
import json
import re
from typing import Optional


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return None


def parse_shopping_trajectory(json_str: str) -> dict:
    """
    {
      "shopping_trajectory": "...."
    }
    """
    m = re.search(
        r'"shopping_trajectory"\s*:\s*"([\s\S]*)"\s*}',
        json_str
    )
    if not m:
        raise ValueError("shopping_trajectory not found")

    value = m.group(1)

    value = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', value)

    value = re.sub(r'(?<!\\)"', r'\\"', value)

    return {"shopping_trajectory": value}


def process_and_write(folder_list, output_path):
    total = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for folder in folder_list:
            for file in os.listdir(folder):
                if not (file.startswith("output_") and file.endswith(".jsonl")):
                    continue

                path = os.path.join(folder, file)
                print(f"Processing: {path}")

                with open(path, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            failed += 1
                            continue

                        resp = row.get("filtered_resps", "")
                        json_str = extract_json_block(resp)
                        if not json_str:
                            continue

                        try:
                            obj = parse_shopping_trajectory(json_str)
                            out.write(
                                json.dumps(obj, ensure_ascii=False) + "\n"
                            )
                            total += 1
                        except Exception:
                            failed += 1

    print(f"\n Written lines: {total}")
    print(f" Failed lines: {failed}")


if __name__ == "__main__":
    folders = [
        '/path/to/data/Pseudo_data/batch/train/splitted_prompts'
    ]

    output_file = "/path/to/data/Pseudo_data/benchmark_data/additional_shopping_trajectory.jsonl"

    process_and_write(folders, output_file)

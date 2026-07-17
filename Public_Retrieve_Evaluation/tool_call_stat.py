import json

def analyze_tool_usage(file_path):
    total = 0
    try_call_tool = 0
    success_call_tool = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1

            preds = data.get("predictions", [])
            if not preds:
                continue

            pred = preds[0]
            code_exec_count = pred.get("code_exec_count", 0)
            succ_code_exec_count = pred.get("succ_code_exec_count", 0)

            if code_exec_count > 0:
                try_call_tool += 1
            if succ_code_exec_count > 0:
                success_call_tool += 1

    result = {
        "total": total,
        "try_call_tool": f"{try_call_tool}/{total}",
        "success_call_tool": f"{success_call_tool}/{total}"
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    # with open(file_path, "a", encoding="utf-8") as f:
    #     f.write("\n" + json.dumps(result, ensure_ascii=False))

# main
if __name__ == "__main__":
    file_path1 = "/home/hongyee/hyenv/phrase2/retrieve_data/rollout/factuality_32k_test_tool_qwen3_4b_sqltool.jsonl"
    file_path2 = "/home/hongyee/hyenv/phrase2/retrieve_data/rollout/factuality_128k_test_tool_qwen3_4b.jsonl"
    file_path3 = "/home/hongyee/hyenv/phrase2/retrieve_data/rollout/factuality_32k_test_tool_qwen3_4b.jsonl"
    file_path4 = "/home/hongyee/hyenv/phrase2/retrieve_data/rollout/factuality_128k_test_tool_qwen3_4b.jsonl"


    analyze_tool_usage(file_path1)
    analyze_tool_usage(file_path2)
    # analyze_tool_usage(file_path3)
    # analyze_tool_usage(file_path4)

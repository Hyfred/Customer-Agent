import os
import re
import json
import argparse
from tqdm import trange
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from executor import PythonExecutor  

def build_prompt(input_text, question):
    
    return (
        "Solve the following problem step by step. You now have the ability to selectively write executable Python code "
        "to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output "
        "(wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive "
        "at the final answer. The Python code should be complete scripts, including necessary imports. \n"
        "You may specifically use Python to:\n"
        "1. Extract or preprocess data (e.g., parse text, identify or clean keywords, categorize items).\n"
        "2. Perform quantitative analysis (e.g., count frequencies, compute ratios, detect time or category trends).\n"
        "Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\n"
        "The last part of your response should be in the following format:\n"
        "<answer>\nYOUR_ANSWER_HERE\n</answer>\n\n"
        "*user question:*\n"
        "Customer purchase history:\n\n"
        f"{input_text}\n\n"
        "Question:\n"
        f"{question}\n"
    )


def extract_answer(output_text):
    # get last <answer>...</answer> content
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

# ======================
# main
# ======================

def main(args, data):
    print(f"=== initial {args.model_name_or_path} ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.parallel_size,
        swap_space=16,
    )

    stop_words = []
    if args.exe_code:
        stop_words.append("</code>")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop_words,
        n=args.n,  
    )
    sampling_params_1 = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop_words,
        n=1,
    )

    executor = PythonExecutor()

    fout = open(args.output_path, "w", encoding="utf-8")

    total_problem, total_correct, missing_pred = 0, 0, 0

    # ==== batch size ====
    for st in trange(0, len(data), args.batch_size, desc="Inference"):
        batch = data[st : st + args.batch_size]
        prompts = [build_prompt(item["trajectory"], item["question"]) for item in batch]
        responses = llm.generate(prompts, sampling_params)

        final_responses, final_code_num_lst, final_succ_code_num_lst = [], [], []

        # ===== do n sample for each data =====
        for response, prompt_text in zip(responses, prompts):
            state_list = []
            for out in response.outputs:  
                state_list.append(
                    {
                        "alive": True,
                        "text": prompt_text,
                        "last_output": out.text,
                        "stop_reason": out.stop_reason,
                        "iter_count": 0,
                        "code_exec_count": 0,
                        "succ_code_exec_count": 0,
                    }
                )

            # === multi-turn ===
            while any(s["alive"] for s in state_list):
                if all(s["iter_count"] >= args.max_iter for s in state_list):
                    for s in state_list:
                        s["alive"] = False
                    break

                # === collect code ===
                code_to_exec, code_indices = [], []
                for idx, s in enumerate(state_list):
                    if not s["alive"]:
                        continue
                    if s["stop_reason"] != "</code>":
                        s["alive"] = False
                        continue
                    code = (
                        s["last_output"].split("```python")[-1].replace("```", "").strip()
                    )
                    if not code:
                        s["alive"] = False
                        continue
                    if s["code_exec_count"] >= args.max_code_exec:
                        s["alive"] = False
                        continue
                    code_to_exec.append(code)
                    code_indices.append(idx)

                if len(code_to_exec) == 0:
                    break

                # === exeute ===
                try:
                    results = executor.batch_apply(code_to_exec)
                except Exception as e:
                    print(f"[WARN] Executor error: {e}")
                    results = [(None, "Error")] * len(code_to_exec)

                for local_i, global_idx in enumerate(code_indices):
                    s = state_list[global_idx]
                    res = results[local_i] if local_i < len(results) else (None, "Error")
                    output, report = res
                    exec_content = output if report == "Done" else report
                    if report == "Done":
                        s["succ_code_exec_count"] += 1

                    s["text"] += (
                        s["last_output"]
                        + "</code>\n<interpreter>\n"
                        + str(exec_content)
                        + "\n</interpreter>\n\n"
                    )
                    s["code_exec_count"] += 1

                # === next step ===
                to_generate = [s["text"] for s in state_list if s["alive"]]
                if len(to_generate) == 0:
                    break

                new_responses = llm.generate(to_generate, sampling_params_1)
                tmp_i = 0
                for s in state_list:
                    if not s["alive"]:
                        continue
                    out = new_responses[tmp_i].outputs.pop(0)
                    tmp_i += 1
                    s["last_output"] = out.text
                    s["stop_reason"] = out.stop_reason
                    s["iter_count"] += 1
                    if s["stop_reason"] != "</code>":
                        s["alive"] = False

            # === concat ===
            fini_responses = [s["text"] + s["last_output"] for s in state_list]
            final_responses.append(fini_responses)
            final_code_num_lst.append([s["code_exec_count"] for s in state_list])
            final_succ_code_num_lst.append([s["succ_code_exec_count"] for s in state_list])

        # ===== eval =====
        for pred_list, item, code_num_lst, succ_code_num_lst in zip(final_responses, batch, final_code_num_lst, final_succ_code_num_lst):
            gt = item["reference_answer"].strip()
            record = {
                "prompt": build_prompt(item["trajectory"], item["question"]),
                "ground_truth": gt,
                "predictions": [],
            }

            for pred_text, code_num, succ_code_num in zip(pred_list, code_num_lst, succ_code_num_lst):
                pred = extract_answer(pred_text)
                is_correct = (pred == gt)
                total_problem += 1
                if not pred:
                    missing_pred += 1
                if is_correct:
                    total_correct += 1

                record["predictions"].append(
                    {
                        "solution": pred_text,
                        "prediction": pred,
                        "correct": is_correct,
                        "code_exec_count": code_num,
                        "succ_code_exec_count": succ_code_num,
                    }
                )

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    # ==== get result ====
    accuracy = total_correct / total_problem if total_problem > 0 else 0
    missing_ratio = missing_pred / total_problem if total_problem > 0 else 0

    summary = {
        "accuracy": round(accuracy * 100, 2),
        "missing_ratio": round(missing_ratio * 100, 2),
        "total": total_problem,
        "correct": total_correct,
    }

    fout.write(json.dumps(summary) + "\n")
    fout.close()
    print("\n=== stat ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"save as: {args.output_path}")


# ======================
# entry
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--parallel_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--exe_code", action="store_true")
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--max_code_exec", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=3, help="sample size")
    args = parser.parse_args()

    # with open(args.input_path, "r", encoding="utf-8") as fin:
    #     data = json.load(fin)
    with open(args.input_path, "r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    print(f"loading {len(data)} data")

    main(args, data)

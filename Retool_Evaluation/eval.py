# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

import os
import re
import json
import argparse
import numpy as np
from tqdm import trange
from evaluator.MC_evaluator_list import MCEvaluator
from evaluator.MATH_evaluator_list import MATHEvaluator
from executor import *

def check(evaluator, pred_ans, real_ans):
    if len(pred_ans) == 0:
        return []
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness

name2path = {
    "AIME24": "dataset/AIME24.jsonl",
    "AIME25": "dataset/AIME25.jsonl",
}

name2eval = {
    "AIME24": MATHEvaluator(),
    "AIME25": MATHEvaluator(),
}


def main(args, lines, start_id, use_slice=False):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # ==== GPU Slice ====
    if use_slice:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            f"{start_id%2*4},{start_id%2*4+1},{start_id%2*4+2},{start_id%2*4+3}"
        )

    # ==== Constants ====
    MAX_ITER_PER_RESPONSE =20
    MAX_CODE_EXECUTIONS_PER_ITEM = 20

    # ==== Init Model & Tokenizer ====
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.paralle_size,
        swap_space=16,
    )

    stop_words = []
    if args.exe_code:
        stop_words.append("</code>")
    executor = PythonExecutor()

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=args.max_tokens,
        stop=stop_words,
        n=args.n,
    )
    sampling_params_1 = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=args.max_tokens,
        stop=stop_words,
        n=1,
    )
    evaluator = name2eval[args.data_name]

    # ==== Helper: Execute Python Codes ====
    def excute_codes(codes, executor: PythonExecutor):
        if len(codes) == 0:
            return []
        try:
            results = executor.batch_apply(codes)
        except Exception as e:
            print(f"[WARN] Executor error: {e}")
            results = [None] * len(codes)
        return results

    # ==== Helper: Build Prompt ====
    def process_prompt(question):
        with open(args.prompt_template, "r") as fin:
            sys = json.load(fin)
        prompt_prefix = sys[args.prompt]
        chat_prob = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_prefix.format(query=question),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return chat_prob

    prefix_tgt = "exe" if args.exe_code else "no_exe"
    tgt_path = os.path.join(
        args.target_path,
        "{}-{}-{}-{}-{}.jsonl".format(
            prefix_tgt,
            args.model_name_or_path.split("/")[-1],
            args.data_name.split("/")[-1],
            args.prompt_template.split("/")[-1].split(".")[0],
            args.n,
        ),
    )
    fout = open(tgt_path, "w")

    bs = 100
    num_data = len(lines)
    total_problem, total_correct = 0, 0

    for st in trange(0, num_data, bs):
        print(f"\n=== Batch start {st}/{num_data} ===")
        tmp_lines = lines[st : st + bs]
        prompts = [process_prompt(data["input"]) for data in tmp_lines]
        responses = model.generate(prompts, sampling_params)

        final_responses, final_code_num_lst = [], []

        # ==== Process each problem ====
        for response, prompt in zip(responses, prompts):
            state = []
            for out in response.outputs:
                state.append({
                    "alive": True,
                    "text": prompt,
                    "last_output": out.text,
                    "stop_reason": out.stop_reason,
                    "iter_count": 0,
                    "code_exec_count": 0,
                })

            # ==== Main loop with safety cap ====
            while any(s["alive"] for s in state):
                if all(s["iter_count"] >= MAX_ITER_PER_RESPONSE for s in state):
                    print("[WARN] Reached MAX_ITER_PER_RESPONSE, force stop.")
                    for s in state:
                        s["alive"] = False
                    break

                # Collect codes to execute
                code_to_exec, code_indices = [], []
                for idx, s in enumerate(state):
                    if not s["alive"]:
                        continue
                    if s["stop_reason"] != "</code>":
                        s["alive"] = False
                        continue
                    code = s["last_output"].split("```python")[-1].replace("```", "").strip()
                    if not code:
                        s["alive"] = False
                        continue
                    if s["code_exec_count"] >= MAX_CODE_EXECUTIONS_PER_ITEM:
                        s["alive"] = False
                        continue
                    code_to_exec.append(code)
                    code_indices.append(idx)

                if len(code_to_exec) == 0:
                    break

                results = excute_codes(code_to_exec, executor)
                for local_i, global_idx in enumerate(code_indices):
                    s = state[global_idx]
                    res = results[local_i] if local_i < len(results) else None
                    if res is None:
                        excu_content = "None"
                    else:
                        output, report = res
                        excu_content = output if report == "Done" else report
                    s["text"] += (
                        s["last_output"]
                        + "</code>\n<interpreter>\n"
                        + excu_content
                        + "</interpreter>\n\n"
                    )
                    s["code_exec_count"] += 1

                # Generate next outputs
                to_generate = [s["text"] for s in state if s["alive"]]
                if len(to_generate) == 0:
                    break
                new_responses = model.generate(to_generate, sampling_params_1)
                tmp_i = 0
                for s in state:
                    if not s["alive"]:
                        continue
                    out = new_responses[tmp_i].outputs.pop(0)
                    tmp_i += 1
                    s["last_output"] = out.text
                    s["stop_reason"] = out.stop_reason
                    s["iter_count"] += 1
                    if s["stop_reason"] != "</code>":
                        s["alive"] = False

            fini_responses = [s["text"] + s["last_output"] for s in state]
            final_responses.append(fini_responses)
            final_code_num_lst.append([s["code_exec_count"] for s in state])

        # ==== Evaluate ====
        for response, data, code_num_lst in zip(final_responses, tmp_lines, final_code_num_lst):
            output_ = data["output"]
            new_data = {
                "input": data["input"],
                "output": output_,
                "prediction": [],
            }
            pred_ans_list, real_ans_list = [], []
            for pred in response:
                pred_ans_list.append(pred)
                real_ans_list.append(output_)

            correctness = check(evaluator, pred_ans_list, real_ans_list)
            pred_last_num_lst = [re.findall(r"\d+", pred_ans.split("\n")[-1]) for pred_ans in pred_ans_list]
            pred_real_pairs = [
                (
                    ("\\boxed{" + nums[-1] + "}", real)
                    if len(nums) > 0 else (False, real)
                )
                for nums, real in zip(pred_last_num_lst, real_ans_list)
            ]
            correctness_last_num_left = check(
                evaluator,
                [c[0] for c in pred_real_pairs if c[0] != False],
                [c[1] for c in pred_real_pairs if c[0] != False],
            )
            correctness_last_num = []
            for idx in range(len(pred_real_pairs)):
                if pred_real_pairs[idx][0] == False:
                    correctness_last_num.append(False)
                else:
                    correctness_last_num.append(correctness_last_num_left.pop(0))
            correctness = [c or c_last for c, c_last in zip(correctness, correctness_last_num)]

            for output, c, code_num in zip(response, correctness, code_num_lst):
                if c:
                    total_correct += 1
                token_len = len(tokenizer.encode(output))
                new_data["prediction"].append({
                    "solution": output,
                    "correctness": c,
                    "token_len": token_len,
                    "code_num": code_num,
                })
                total_problem += 1
            fout.write(json.dumps(new_data) + "\n")
            fout.flush()

    results = {"accuracy": round(total_correct / total_problem * 100, 2)}
    fout.write(json.dumps(results) + "\n")
    fout.flush()
    fout.close()
    print(f"Accuracy: {results['accuracy']}% ({total_correct}/{total_problem})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--max_tokens", default=1000, type=int)
    parser.add_argument("--paralle_size", default=8, type=int)
    parser.add_argument("--year", default=None, type=str, required=False)
    parser.add_argument("--prompt", default="r1_code", type=str, required=False)
    parser.add_argument("--decode", default="sample", type=str)
    parser.add_argument("--use_slice", action="store_true")
    parser.add_argument("--slice_id", default=0, type=int)
    parser.add_argument("--prompt_template", default=None, type=str)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--exe_code", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.target_path, exist_ok=True)
    src_path = name2path[args.data_name]

    with open(src_path, "r") as fin:
        dataset = [json.loads(d) for d in fin.readlines()]
    print("Total data:", len(dataset))

    if args.use_slice:
        slice_idx = np.linspace(0, len(dataset), 3).astype("int")
        start, end = slice_idx[args.slice_id], slice_idx[args.slice_id + 1]
        dataset = dataset[start:end]
        print(f"Start slice {args.slice_id} from {start} to {end}")

    main(args, dataset, args.slice_id, args.use_slice)
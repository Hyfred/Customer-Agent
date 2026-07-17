import os
import re
import json
import argparse
from tqdm import trange
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from executor import PythonExecutor  
import botocore.config
import boto3

from openai import OpenAI



def build_prompt(qid, question):
    return (
        "<|im_start|>system\n"
        "# Tools\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        "{\n"
        "  \"type\": \"function\",\n"
        "  \"function\": {\n"
        "    \"name\": \"code_interpreter\",\n"
        "    \"description\": \"A tool for executing code.\",\n"
        "    \"parameters\": {\n"
        "      \"type\": \"object\",\n"
        "      \"properties\": {\n"
        "        \"code\": {\n"
        "          \"type\": \"string\",\n"
        "          \"description\": \"The code to execute.\"\n"
        "        }\n"
        "      },\n"
        "      \"required\": [\"code\"]\n"
        "    }\n"
        "  }\n"
        "}\n"
        "</tools>\n\n"

        "For each function call, return a JSON object enclosed in <tool_call></tool_call> tags.\n"
        "Example:\n"
        "<tool_call>\n"
        "{ \"name\": \"code_interpreter\", \"arguments\": { \"code\": \"print('hello')\" } }\n"
        "</tool_call>\n"
        "<|im_end|>\n"

        "<|im_start|>user\n"
        "You are a Python reasoning assistant that retrieves answers using the provided question and its corresponding qid.\n"
        "The qid helps you locate the correct data file. Load it and answer the question.\n"
        "You must always use Python for reasoning and answering.\n\n"

        "### Data Access\n"
        "- Do NOT manually read or parse files.\n"
        "- Always load data using:\n"
        "  from data_loader import load_trajectory\n"
        f"  conn, df = load_trajectory('{qid}')\n"
        "- `df` is a pandas DataFrame for convenience, **but the SQLite table is always named `events`**.\n"
        "- Do NOT attempt to query a table named `df`.\n"
        "- The `events` table has columns: timestamp, action_type, asin, product_name, brand, product_type, price, search_query, ...\n\n"

        "### SQL Query\n"
        "Use:\n"
        "  from data_loader import sql_query\n"
        "  validate_true, sql_result = sql_query(conn, <SQL>)\n\n"

        "### Instructions\n"
        f"1. Load trajectory with `load_trajectory('{qid}')`.\n"
        "2. Always print intermediate steps: df.head(), df.columns, and SQL results.\n"
        "3. Translate the question into an SQL query.\n"
        "4. Execute using sql_query().\n\n"

        "### Output Rules\n"
        "- Every code block must be complete and runnable independently.\n"
        "- Print intermediate parsing details, dataframe summary, and query results.\n\n"

        f"*User question:*\n"
        f"Trajectory file: trajectory_files/trajectory_{qid}.txt\n"
        f"Question: {question}\n\n"

        "The answer must be formatted as:\n"
        "\\boxed{'The final answer goes here.'}\n"
        "<|im_end|>\n"

        "<|im_start|>assistant\n"
    )


def extract_answer(output_text):

    # match \boxed{...}
    pattern = r"\\boxed\s*\{\s*(['\"]?)(.*?)\1\s*\}"

    matches = re.findall(pattern, output_text, flags=re.S)

    if not matches:
        return None

    content = matches[-1][1].strip()

    content = content.strip("`'\" ")

    return content


def extract_python_code(output_text):

    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    
    start = output_text.find(start_tag)
    # end = output_text.find(end_tag)
    if start == -1:
        return None

    json_text = output_text[start + len(start_tag): -1].strip()


    first_brace = json_text.find("{")
    if first_brace == -1:
        return None

    i = first_brace
    brace_count = 0
    in_string = False
    escape = False

    for j in range(first_brace, len(json_text)):
        c = json_text[j]

        if escape:
            escape = False
            continue

        if c == "\\":
            escape = True
            continue

        if c in ['"', "'"]:
            if not in_string:
                in_string = c
            elif in_string == c:
                in_string = False
            continue

        if not in_string:
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_obj_str = json_text[first_brace : j + 1]
                    break
    else:
        return None
    
    try:
        call_obj = json.loads(json_obj_str)
    except Exception as e:
        print("JSON parse error:", e)
        return None

    return call_obj.get("arguments", {}).get("code")


def save_trajectory_file(qid, trajectory, folder="trajectory_files"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"trajectory_{qid}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(trajectory)
    return file_path

from types import SimpleNamespace

def init_openai_client():
    return OpenAI(
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
        api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    )

def openai_generate(
    client,
    prompts,
    temperature,
    top_p,
    max_tokens,
    n=1,
):
    model_id = "openai.gpt-oss-120b-1:0"
    responses = []

    for prompt in prompts:
        outputs = []

        for _ in range(n):
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "developer", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

                text = completion.choices[0].message.content
                stop_reason = completion.choices[0].finish_reason

            except Exception as e:
                text = ""
                stop_reason = f"error: {e}"

            outputs.append(
                SimpleNamespace(
                    text=text,
                    stop_reason=stop_reason,
                )
            )

        responses.append(SimpleNamespace(outputs=outputs))

    return responses

def bedrock_generate(
    client,
    prompts,
    temperature,
    top_p,
    max_tokens,
    n=1,
):
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    responses = []

    for prompt in prompts:
        outputs = []

        for _ in range(n):
            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[
                        {"role": "user", "content": [{"text": prompt}]}
                    ],
                    inferenceConfig={
                        "temperature": temperature,
                        "topP": top_p,
                        "maxTokens": max_tokens,
                    },
                )

                text = resp["output"]["message"]["content"][0]["text"]
                stop_reason = resp.get("stopReason", "end")

            except Exception as e:
                text = ""
                stop_reason = f"error: {e}"

            outputs.append(SimpleNamespace(text=text, stop_reason=stop_reason))

        responses.append(SimpleNamespace(outputs=outputs))

    return responses

def api_generate(
    api_type,
    client,
    prompts,
    temperature,
    top_p,
    max_tokens,
    n,
):
    if api_type == "openai":
        return openai_generate(
            client, prompts,
            temperature, top_p, max_tokens, n
        )
    else:
        return bedrock_generate(
            client, prompts,
            temperature, top_p, max_tokens, n
        )


# ======================
# main function
# ======================

def main(args, data):
    print(f"=== initialize {args.api_type} ===")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    # llm = LLM(
    #     model=args.model_name_or_path,
    #     trust_remote_code=True,
    #     dtype="bfloat16",
    #     tensor_parallel_size=args.parallel_size,
    #     swap_space=16,
    # )

    stop_words = []
    if args.exe_code:
        # stop_words.append("</code>")
        stop_words.append("</tool_call>")
    else:
        stop_words.append("<|im_end|>")



    executor = PythonExecutor()

    fout = open(args.output_path, "w", encoding="utf-8")

    total_problem, total_correct, missing_pred = 0, 0, 0

    # ==== batch ====
    for st in trange(0, len(data), args.batch_size, desc="Inference"):
        batch = data[st : st + args.batch_size]
        # prompts = [build_prompt(item["trajectory"], item["question"]) for item in batch]
        prompts = []
        for item in batch:
            # save trajectory in local
            save_trajectory_file(item["qid"], item["trajectory"])
            # build prompt
            prompts.append(build_prompt(item["qid"],item["question"]))

        # responses = llm.generate(prompts, sampling_params)
        responses = api_generate(
            api_type=args.api_type,
            client=client,
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n,
        )

        final_responses, final_code_num_lst, final_succ_code_num_lst = [], [], []

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

            # === multi turn ===
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

                    # detect <tool_call>
                    code = extract_python_code(s["last_output"])

                    if code is None:  # if not code call will not execute
                        s["alive"] = False
                        continue

                    if s["code_exec_count"] >= args.max_code_exec:
                        s["alive"] = False
                        continue

                    code_to_exec.append(code)
                    code_indices.append(idx)

                if len(code_to_exec) == 0:
                    break

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
                        + "user\n"
                        + "<tool_response>\n"
                        + str(exec_content)
                        + "\n</tool_response>\n"
                        + "assistant\n"
                    )
                    s["code_exec_count"] += 1

                to_generate = [s["text"] for s in state_list if s["alive"]]
                if len(to_generate) == 0:
                    break


                # max_len = getattr(llm.llm_engine.model_config, "max_model_len", 40960)
                max_len = 40960
                reserve_tokens = 512  
                safe_len = max_len - reserve_tokens

                def safe_truncate(prompt, tokenizer, safe_len):
                    tokens = tokenizer.encode(prompt)
                    if len(tokens) > safe_len:
                        print(f"[Warning] Truncating prompt from {len(tokens)} → {safe_len} tokens.")
                        tokens = tokens[:safe_len]
                        prompt = tokenizer.decode(tokens)
                    return prompt

                to_generate = [safe_truncate(p, tokenizer, safe_len) for p in to_generate]

                # new_responses = llm.generate(to_generate, sampling_params_1)
                new_responses = api_generate(
                    api_type=args.api_type,
                    client=client,
                    prompts=to_generate,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    n=1,
                )
                tmp_i = 0
                for s in state_list:
                    if not s["alive"]:
                        continue
                    out = new_responses[tmp_i].outputs.pop(0)
                    tmp_i += 1
                    s["last_output"] = out.text
                    s["stop_reason"] = out.stop_reason
                    s["iter_count"] += 1
                    if extract_python_code(out.text) is None:
                        s["alive"] = False

            fini_responses = [s["text"] + s["last_output"] for s in state_list]
            final_responses.append(fini_responses)
            final_code_num_lst.append([s["code_exec_count"] for s in state_list])
            final_succ_code_num_lst.append([s["succ_code_exec_count"] for s in state_list])

        for pred_list, item, code_num_lst, succ_code_num_lst in zip(final_responses, batch, final_code_num_lst, final_succ_code_num_lst):
            gt = item["reference_answer"].strip()
            record = {
                "prompt": build_prompt(item["qid"],item["question"]),
                "ground_truth": gt,
                "question": item["question"],
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

    # ==== stat ====
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
    print(f"result save as: {args.output_path}")



# ======================
# main
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--api_type", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--exe_code", action="store_true")
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--max_code_exec", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=3, help="sample size")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    print(f"Loading {len(data)} data")

    # ====== configuration ======
    # Set AWS_BEARER_TOKEN_BEDROCK in your environment before running:
    # export AWS_BEARER_TOKEN_BEDROCK="<your-bedrock-api-key>"

    config = botocore.config.Config(
        read_timeout=400,      
        connect_timeout=90,    
    )

    if args.api_type == "openai":
        client = init_openai_client()
    else:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            config=config
        )

    main(args, data)


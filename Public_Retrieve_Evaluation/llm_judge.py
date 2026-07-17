import json
import argparse
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, Any, List


# def build_judge_prompt(question: str, reference_answer: str, model_response: str) -> str:
#     judge_prompt = f'''You are an expert evaluator. Your task is to determine if a model's answer is factually correct for a given question.
    
# Question: {question}
# Expected Answer: {reference_answer}
# Model Answer: {model_response}

# Evaluate whether the model's answer is factually correct. Consider:
# 1. Semantic equivalence (different wording but same meaning)
# 2. Format differences are acceptable (e.g., "5" vs "five", "2024-12-14" vs "Dec 14, 2024")
# 3. Numerical precision (close enough values)
# 4. Product names (reasonably close brand/product names)
# 5. Answer format flexibility: If the question asks for a product but the model gives the price, that's still correct if the price matches the expected product
# 6. If the model provides the correct factual information, even with additional context, mark as CORRECT
# 7. If the model provides incorrect factual information or contradicts the expected answer, mark as INCORRECT
# 8. If the `prediction` is different, or contains only placeholders (e.g., "final_answer", "PLACEHOLDER_FOR_ANSWER") or indicates no useful information (e.g., "No search event found"), mark as INCORRECT.
# 9. Do NOT assume any answer or invent reasoning. Only the content in `Model Answer` matters.

# Be objective and factual in your evaluation. The answer must align with the expected factual information.

# Output requirements:
# - First show your analysis.
# - Final line MUST be:
#   {{"answer": "CORRECT"}}
#   or
#   {{"answer": "INCORRECT"}}
# - Do not include any extra text after this JSON.'''
#     return judge_prompt


def build_judge_prompt(question: str, reference_answer: str, model_response: str) -> str: 
    judge_prompt = f'''You are an expert evaluator. Your task is to determine if a model's answer is factually correct for a given question. 
    Question: {question} Expected Answer: {reference_answer} Model Answer: {model_response} 
    - Compare the Model Answer with the Expected Answer. 
    - If the Model Answer conveys the correct factual information (semantic equivalence), even with different wording or formatting, mark as CORRECT. 
    - If the Model Answer is factually incorrect, contradicts the Expected Answer, or contains only placeholders (e.g., "PLACEHOLDER_FOR_ANSWER", "final_answer"), "None", or indicates no useful information (e.g., "No search event found"), mark as INCORRECT. 
    - Do NOT assume, invent, or speculate about missing information. Only evaluate the actual content provided in the Model Answer. 
    - Output ONLY one JSON object as the final line: {{"answer": "CORRECT"}} or {{"answer": "INCORRECT"}}.'''
    return judge_prompt

def build_judge_prompt(question: str, reference_answer: str, model_response: str) -> str:
    judge_prompt = f'''You are an expert evaluator. Your task is to determine if a model's answer is factually correct for a given question.

Question: {question}
Expected Answer: {reference_answer}
Model Answer: {model_response}

- Compare the Model Answer with the Expected Answer.
- Minor differences in wording, formatting, or numeric representation are acceptable if the factual meaning is preserved.
- Product names (reasonably close brand/product names) are acceptable.
- Answer format flexibility: If the question asks for a product but the model gives the price, that's still correct if the price matches the expected product.
- If the Model Answer conveys the correct factual information (semantic equivalence), mark as CORRECT.
- If the Model Answer is factually incorrect, contradicts the Expected Answer, or contains only placeholders (e.g., "PLACEHOLDER_FOR_ANSWER", "{{final_answer}}"), "None", or indicates no useful information (e.g., "No search event found"), mark as INCORRECT.
- Do NOT assume, invent, or speculate about missing information. Only evaluate the actual content provided in the Model Answer.
- Output ONLY one JSON object on the final line, and nothing else: {{"answer": "CORRECT"}} or {{"answer": "INCORRECT"}}.
'''
    return judge_prompt

# def extract_json_from_text(text: str) -> Dict[str, Any]:
#     """Extract a JSON object from judge response using robust multi-step strategies."""
#     if not text or not text.strip():
#         return {}

#     def try_parse(s: str):
#         s = s.strip()
#         try:
#             parsed = json.loads(s)
#             return parsed if isinstance(parsed, dict) else None
#         except Exception:
#             return None

#     # --- Strategy 1: Direct parse ---
#     direct = try_parse(text)
#     if direct:
#         return direct

#     # --- Strategy 2: Extract JSON inside ```json ... ``` blocks ---
#     code_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
#     for block in code_blocks:
#         parsed = try_parse(block)
#         if parsed:
#             return parsed

#     # --- Strategy 3: Extract any {...} block using regex (non-greedy) ---
#     candidates = re.findall(r"\{[\s\S]*?\}", text)
#     best = None
#     for c in candidates:
#         parsed = try_parse(c)
#         if parsed:
#             if "answer" in parsed:
#                 return parsed
#             best = parsed

#     if best:
#         return best

#     # --- Strategy 4: Manual brace matching as fallback ---
#     start = text.find("{")
#     if start != -1:
#         brace = 0
#         in_str = False
#         escape = False

#         for i in range(start, len(text)):
#             ch = text[i]
#             if escape:
#                 escape = False
#                 continue
#             if ch == "\\":
#                 escape = True
#                 continue
#             if ch == '"':
#                 in_str = not in_str
#                 continue

#             if not in_str:
#                 if ch == "{":
#                     brace += 1
#                 elif ch == "}":
#                     brace -= 1
#                     if brace == 0:
#                         json_str = text[start:i + 1]
#                         parsed = try_parse(json_str)
#                         if parsed:
#                             return parsed

#     return {}

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract the last JSON object from text using robust multi-step strategies."""
    if not text or not text.strip():
        return {}

    def try_parse(s: str):
        s = s.strip()
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    # --- Strategy 1: Direct parse ---
    direct = try_parse(text)
    if direct:
        return direct  # if entire text is a JSON, return it

    # --- Strategy 2: Extract JSON inside ```json ... ``` blocks ---
    code_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in reversed(code_blocks):  # start from last block
        parsed = try_parse(block)
        if parsed:
            return parsed

    # --- Strategy 3: Extract any {...} block using regex (non-greedy) ---
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for c in reversed(candidates):  # iterate from last to first
        parsed = try_parse(c)
        if parsed:
            return parsed

    # --- Strategy 4: Manual brace matching as fallback (last {...} block) ---
    start_indices = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(start_indices):  # check from last '{'
        brace = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue

            if not in_str:
                if ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        json_str = text[start:i + 1]
                        parsed = try_parse(json_str)
                        if parsed:
                            return parsed
                        break  # stop this block

    return {}

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Process judge evaluation results for a single sample.
    """
    if not results:
        return {"judge_accuracy": -1.0}

    judge_response = results[0]

    parsed = extract_json_from_text(judge_response)

    if isinstance(parsed, dict) and "answer" in parsed:
        answer = parsed["answer"].strip().upper()
        if answer == "CORRECT":
            return {"judge_accuracy": 1.0}
        elif answer == "INCORRECT":
            return {"judge_accuracy": 0.0}

    return {"judge_accuracy": -1.0}


def main(args):
    file_path = args.file_path

    # ===== 1. Initialize model =====
    print(f"Loading model from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.parallel_size,
        swap_space=16,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
        # presence_penalty=0.5,      
        # frequency_penalty=0.1,     
        repetition_penalty=1.2,    
    )

    # ===== 2. Read data =====
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if lines and lines[-1].strip().startswith('{"accuracy"'):
        last_stats_line = lines[-1]
        data_lines = lines[:-1]
    else:
        last_stats_line = None
        data_lines = lines

    results = []
    total = 0
    correct = 0
    unknown = 0

    prompts = []
    meta_list = []

    # ===== 3. Build judge prompts =====
    data_lines = [line for line in data_lines if '"predictions":' in line]

    for line_idx, line in enumerate(data_lines):
        data = json.loads(line)
        question = data.get("question", "")
        reference_answer = data.get("ground_truth", "")

        for pred_idx, p in enumerate(data["predictions"]):
            model_response = p["prediction"]
            prompt = build_judge_prompt(question, reference_answer, model_response)
            prompts.append(prompt)
            meta_list.append((line_idx, pred_idx))

    # ===== 4. Run batch inference =====
    print(f"Running LLM judgments on {len(prompts)} pairs ...")
    outputs = llm.generate(prompts, sampling_params)

    data_objs = [json.loads(l) for l in data_lines]

    # ===== 5. Process results =====
    for out, (line_idx, pred_idx) in zip(outputs, meta_list):
        ans_text = out.outputs[0].text.strip()
        metrics = process_results({}, [ans_text])
        judge_accuracy = metrics["judge_accuracy"]

        data_objs[line_idx]["predictions"][pred_idx]["llm_judge_text"] = ans_text

        if judge_accuracy == 1.0:
            data_objs[line_idx]["predictions"][pred_idx]["llm_judge"] = True
            correct += 1
        elif judge_accuracy == 0.0:
            data_objs[line_idx]["predictions"][pred_idx]["llm_judge"] = False
        else:
            data_objs[line_idx]["predictions"][pred_idx]["llm_judge"] = None
            unknown += 1

        total += 1

    # ===== 6. Write updated file =====
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n LLM Judge Accuracy = {accuracy:.2f}%, Total = {total}, Correct = {correct}, Unknown = {unknown}")

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_objs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if last_stats_line:
            f.write(last_stats_line)
        f.write(json.dumps({
            "llm_accuracy": round(accuracy, 2),
            "total": total,
            "correct": correct,
            "unknown": unknown
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the jsonl file")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path or name of the Qwen model")
    parser.add_argument("--parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    main(args)

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
from typing import Any

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op
# from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
import time

# set_endpoint("http://localhost:8070")


logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "yentinglin/aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",#tool_agent,single_turn_agent
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"#tool_agent,single_turn_agent
        # qid = row["extra_info"]["qid"]
        # trajectory = row["extra_info"]["trajectory"]

        # save_traj_code = f'''from data_loader import save_trajectory_file
        # save_trajectory_file({qid}, {trajectory})
        # print(1)'''

        # run_code(RunCodeRequest(code=save_traj_code, language='python'),
        #         max_attempts=1,
        #         client_timeout=3)
        # time.sleep(1)   

        return row


# def compute_score(data_source, solution_str, ground_truth, extra_info):
#     # use \\boxed{...} answer
#     result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

#     # breakpoint()
#     # encourage model to call tools
#     num_turns = extra_info["num_turns"]
#     if result["score"] < 0:
#         tool_call_reward = (num_turns - 2) / 2 * 0.1
#         result["score"] = min(-0.6, result["score"] + tool_call_reward)

#     if result["pred"] is None:
#         result["pred"] = ""

#     return result


import numpy as np
import json
import os
import traceback

# this would save the rollout
LOG_JSONL_PATH = os.path.expanduser("/workspace/hyenv/neo_rlvr_training/qwen3-1.7b-instruct-rlvr-neo_ro.jsonl")


def make_json_safe(obj):
    """Recursively convert numpy/int64/float32 objects to JSON-safe Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj


def log_to_jsonl(record, path=LOG_JSONL_PATH):
    """Append one JSON record to .jsonl file (safe append mode)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            json.dump(make_json_safe(record), f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        print(f"[WARN] Failed to log jsonl: {e}")


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Compute model reward score with safe JSON serialization and log key info.
    """
    try:
        result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
        result = make_json_safe(result)

        # get extra_info info
        num_turns = int(extra_info.get("num_turns", 0))
        raw_problem = extra_info.get("raw_problem", "")

        # tool call reward
        if result.get("score", 0) < 0:
            tool_call_reward = max(0, (num_turns - 2) / 2 * 0.1)
            result["score"] = min(-0.6, result["score"] + tool_call_reward)

        if result.get("pred") is None:
            result["pred"] = ""

        # build a log
        log_record = {
            "data_source": data_source,
            "raw_problem": raw_problem,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "num_turns": num_turns,
            "score": result.get("score", None),
            "pred": result.get("pred", None),
        }

        # write to jsonl rollout
        log_to_jsonl(log_record)

        # breakpoint()
        return result

    except Exception as e:
        print(f"[ERROR] compute_score failed: {e}")
        traceback.print_exc()
        return {"score": 0.0, "pred": "", "error": str(e)}
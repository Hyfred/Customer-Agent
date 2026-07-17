# import os
# import time
# import json
# import boto3
# import botocore.config


# def robust_call_gpt_oss_api(
#     prompt: str,
#     model_id: str = "openai/gpt-oss-120b",
#     max_retries: int = 3
# ) -> str:
#     """Call GPT-OSS API with robust error handling and JSON enforcement."""

#     # export AWS_BEARER_TOKEN_BEDROCK=xxxx
#     os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")

#     config = botocore.config.Config(
#         read_timeout=400,
#         connect_timeout=90,
#     )

#     client = boto3.client(
#         service_name="bedrock-runtime",
#         region_name="us-east-1",
#         config=config
#     )

#     # Bedrock OpenAI GPT-OSS model
#     model_id = "us.openai.gpt-oss-120b-1:0"

#     system_prompt = """You are a JSON generation expert. Your ONLY job is to return valid JSON arrays.

# CRITICAL RULES:
# 1. Return ONLY the JSON array - no explanations, no reasoning, no markdown fences
# 2. Do not include any text before or after the JSON
# 3. Ensure the JSON is properly formatted and valid
# 4. If you cannot generate valid JSON, return an empty array: []

# Example of correct output:
# [{"question": "What was the first product?", "sql": "SELECT * FROM events LIMIT 1", "answer": "Product A", "difficulty": "Easy"}]

# Return ONLY the JSON array now:"""

#     for attempt in range(max_retries):
#         try:
#             response = client.converse(
#                 modelId=model_id,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"text": system_prompt},
#                             {"text": prompt}
#                         ]
#                     }
#                 ],
#                 inferenceConfig={
#                     "maxTokens": 2048,
#                     "temperature": 0.1,
#                     "topP": 0.9
#                 }
#             )

#             return response["output"]["message"]["content"][0]["text"]

#         except Exception as e:
#             print(f"[Attempt {attempt + 1}] Bedrock call failed: {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(2)
#             else:
#                 return ""


# def test_robust_call():
#     print("=== Bedrock GPT-OSS API Test ===")

#     test_prompt = (
#         "Generate 2 SQL questions about user login events. "
#         "Each item must include question, sql, answer, difficulty."
#     )

#     result = robust_call_gpt_oss_api(test_prompt)

#     print("\n--- Raw Model Output ---")
#     print(result)

#     print("\n--- JSON Validation ---")
#     try:
#         parsed = json.loads(result)
#         assert isinstance(parsed, list)
#         print(" Valid JSON array")
#         print(json.dumps(parsed, indent=2, ensure_ascii=False))
#     except Exception as e:
#         print(" Invalid JSON output")
#         print(e)


# if __name__ == "__main__":
#     test_robust_call()

from openai import OpenAI

client = OpenAI(
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", 
    api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "") # Replace with actual API key
)

completion = client.chat.completions.create(
    model="openai.gpt-oss-120b-1:0",
    messages=[
        {
            "role": "developer",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello! I hate"
        }
    ]
)

print(completion.choices[0].message)
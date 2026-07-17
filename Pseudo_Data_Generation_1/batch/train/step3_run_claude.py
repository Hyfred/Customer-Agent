import json
import boto3
import os
from tqdm import tqdm
import time
import botocore.config


# ====== configuration ======
input_path = "batch3/train/splitted_prompts/prompts_2000_300_dec11_longer_action_part_60.jsonl"        
output_path = "batch3/train/splitted_prompts/output_60.jsonl"     
# remeber to change the key below
# Set AWS_BEARER_TOKEN_BEDROCK in your environment before running:
# export AWS_BEARER_TOKEN_BEDROCK="<your-bedrock-api-key>"

config = botocore.config.Config(
    read_timeout=400,      # wait model for 300 
    connect_timeout=90,    # wait connection for 60 
)

client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    config=config
)

model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    error_count=0
    for line in tqdm(fin, desc="Processing"):
        row = json.loads(line)
        
        prompt = row["Question"]   

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]

        # API calling
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
            )

            output_text = response['output']['message']['content'][0]['text']

            # time.sleep(30)

        except Exception as e:
            error_count +=1
            output_text = f"ERROR: {str(e)}"

        # write JSONL
        fout.write(json.dumps(
            {
                # "Question": prompt,
                "filtered_resps": output_text
            },
            ensure_ascii=False
        ) + "\n")
print(error_count)
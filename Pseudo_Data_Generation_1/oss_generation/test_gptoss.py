from openai import OpenAI

client = OpenAI(
    # base_url="http://<your-vllm-host>:8000/v1",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

result = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what tennis grand slam is."}
    ],
    temperature=1.0, 
    reasoning_effort="high"
)
print(result.choices[0].message.content)
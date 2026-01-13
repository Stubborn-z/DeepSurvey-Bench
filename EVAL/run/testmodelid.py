from openai import OpenAI
api_key = "sk-xxx"
client = OpenAI(api_key=api_key, base_url="https://api.ai-gaochao.cn/v1")
# api_key = "sk-xxx"
# client = OpenAI(api_key=api_key, base_url="https://apix.ai-gaochao.cn")

completion = client.chat.completions.create(
model="gpt-5",
messages=[
{"role": "system", "content": ""},
{"role": "user", "content": "解释一下什么是CUDA."}
],
temperature=0.0
)

print(completion.choices[0].message)
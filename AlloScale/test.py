import openai
import random
import string
import requests

port = 30000
client = openai.Client(
    base_url=f"http://127.0.0.1:{port}/v1",
    api_key="None"
)

NUM_REQUESTS = 410
INPUT_LEN = 100  # 约等于 100 tokens（粗略）


def build_random_prompt(length, seed):
    random.seed(seed)
    chars = string.ascii_letters + string.digits + "测试性能分析"
    return "".join(random.choices(chars, k=length))


def send(i, prompt):
    response = client.chat.completions.create(
        model="../deepseek-ai/DeepSeek-V2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1,
    )
    return response


def flush_cache():
    url = f"http://localhost:{port}/flush_cache"
    response = requests.post(url)
    print(response.text)


def preheat_model():
    prompt = "预热一下模型"
    send(-1, prompt)
    print("[warmup] done")


def run_stress_test():
    print(f"\n=== Running {NUM_REQUESTS} requests (len≈{INPUT_LEN}) ===")

    for i in range(NUM_REQUESTS):
        prompt = build_random_prompt(INPUT_LEN, seed=i)

        resp = send(i, prompt)

        print(f"[req {i:03d}] len={len(prompt)} OK")


if __name__ == "__main__":
    flush_cache()
    # preheat_model()
    # run_stress_test()
    # flush_cache()

import asyncio
import openai

port = 30000
client = openai.AsyncClient(
    base_url=f"http://127.0.0.1:{port}/v1",
    api_key="None"
)

async def send(i):
    response = await client.chat.completions.create(
        model="../deepseek-ai/DeepSeek-V2",
        messages=[{"role": "user", "content": f"你好"}],
        temperature=0,
        max_tokens=1,
    )
    return response.choices[0].message.content

async def main():
    tasks = [send(i) for i in range(1)]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())

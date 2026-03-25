import argparse
import time
from typing import Tuple
import aiohttp
import asyncio
import json


async def sglang_inference_call_server(prompt, in_num, out_num, sampled_in_num, sampled_out_num, sleep_time, config,
                                       logger, event_id):
    await asyncio.sleep(sleep_time)

    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    url = f"http://{config.server_config['host']}:{config.server_config['port']}/v1/chat/completions"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "model": config.server_config.get("model", "default"),  # SGLang 也需要指定 model
            "messages": [{"role": "user", "content": prompt}],
            "stream": config.server_config['stream'],
            "max_tokens": int(out_num),
            "temperature": config.server_config['temperature'],
        }

        first_chunk_time = 0
        start_time = time.perf_counter()
        full_content = ""

        async with session.post(url, json=generation_input) as resp:
            if resp.status != 200:
                print(f"SGLang Error: {resp.status} {resp.reason}")
                return None

            if config.server_config['stream']:
                first_chunk_received = False
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        if not first_chunk_received:
                            first_chunk_time = time.perf_counter() - start_time
                            first_chunk_received = True

                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                full_content += delta["content"]
                        except Exception:
                            continue
            else:
                output = await resp.json()
                full_content = output["choices"][0]["message"]["content"]

            end_time = time.perf_counter()
            total_chunk_time = end_time - start_time

            logger.tick_end(event_id, time.perf_counter())

            save_query_json = {
                "event_id": event_id,
                "out_len": len(full_content),
                "out_len_expected": int(out_num),
                "in_len": int(in_num),
                "sampled_in_num": int(sampled_in_num),
                "sampled_out_len": int(sampled_out_num),
                "first_chunk_time": first_chunk_time,
                "total_chunk_time": total_chunk_time,
                "record_time": time.perf_counter()
            }
            with open(logger.log_path, "a") as f:
                f.write("\n")
                json.dump(save_query_json, f)

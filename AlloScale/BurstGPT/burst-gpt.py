import sys
import os
import yaml
from utils.server import ServerOnline
from utils.config import Config

if __name__ == "__main__":
    with open("profile-config.yaml", 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    server_config = full_config.get('server', {})
    prompt_config = full_config.get('prompt', {})

    print(f"Prompt Config: {prompt_config}")
    config = Config(server_config=server_config, prompt_config=prompt_config)

    server = ServerOnline(
        model_path=full_config.get('model_path'),
        data_path=full_config.get('data_path', "loader/shareGPT.json"),
        backend="sglang",
        log_path=full_config.get('log_path', "server_log.json"),
        config=config,
        detail_log_path=full_config.get('detail_log_path', "detail_server_log.json")
    )

    server.start_profile()
    server.save_log()

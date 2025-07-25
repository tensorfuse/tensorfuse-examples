import os
import sys
import json
import subprocess
import yaml
import pprint
from tensorkube import get_queued_message

def recursive_update(original, override):
    for key, value in override.items():
        if (
            key in original
            and isinstance(original[key], dict)
            and isinstance(value, dict)
        ):
            original[key] = recursive_update(original[key], value)
        else:
            original[key] = value
    return original

def run_finetuning(message_json):
    print("starting training job")

    # 1. Load original YAML config
    with open("axolotl-config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)

    # 2. Parse the message (already a dict or needs json.loads)
    if isinstance(message_json, str):
        json_message = json.loads(message_json)
    else:
        json_message = message_json

    # 3. Override YAML config with JSON
    updated_config = recursive_update(yaml_config, json_message)

    # 4. Write new YAML config
    with open("axolotl-config.yaml", "w") as f:
        yaml.safe_dump(updated_config, f, sort_keys=False)

    # 5. Launch training
    result = subprocess.run(
        ['accelerate', 'launch', '-m', 'axolotl.cli.train', 'axolotl-config.yaml'],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ
    )
    print("training job completed")
    pprint.pprint(result)

if __name__ == "__main__":
    # Example: message = get_queued_message()
    message = get_queued_message()  # returns a JSON string
    run_finetuning(message)

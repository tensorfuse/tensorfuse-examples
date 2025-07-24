import os
import sys
import json
import subprocess
import yaml
from huggingface_hub import HfApi
import pprint



def run_finetuning(message):
    print("starting training job")
    # json_message = json.loads(message)
    # repo_id = json_message.pop('repo_id', None)
    # yaml_message = yaml.dump(json_message, sort_keys=False) # convert json to yaml
    # with open('axolotl-config.yaml', 'w') as file:          # write yaml to file
    #     file.write(yaml_message)
    
    result = subprocess.run(['accelerate', 'launch', '-m', 'axolotl.cli.train', 'axolotl-config.yaml' ], check=True, stdout=sys.stdout, stderr=sys.stderr, env=os.environ) # run training job
    print("training job completed")

    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", None)
    pprint.pprint(result)

    repo_id = 'test_qwen3_finetuning'

if __name__ == "__main__":
    # message = get_queued_message()
    run_finetuning({})
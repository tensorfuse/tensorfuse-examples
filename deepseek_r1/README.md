## Deploying Deepseek-R1-671B with Tensorfuse

Each Tensorkube deployment requires:
1. **Your code** (in this example, vLLM API server code is used from the Docker image).
2. **Your environment** (as a Dockerfile).
3. **A deployment configuration** (`deployment.yaml`).

We will also add **token-based authentication** to our service, compatible with OpenAI client libraries. We will store the authentication token (`VLLM_API_KEY`) as a [Tensorfuse secret](/concepts/secrets). Unlike some other models, **Deepseek-R1 671B** does not require a separate Hugging Face token, so we can skip that step.

### Step 1: Set your API authentication token
Generate a random string that will be used as your API authentication token. Store it as a secret in Tensorfuse using the command below. For the purpose of this demo, we will be using `vllm-key` as your API key.

```bash
tensorkube secret create vllm-token VLLM_API_KEY=vllm-key --env default
```

Ensure that in production you use a randomly generated token. You can quickly generate one
using `openssl rand -base64 32` and remember to keep it safe as [Tensorfuse secrets](/concepts/secrets) are opaque.

### Step 2: Prepare the Dockerfile

We will use the official vLLM Openai image as our base image. This image comes with all the necessary
dependencies to run vLLM. The image is present on DockerHub as [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags).

```dockerfile Dockerfile

# Dockerfile for Deepseek-R1-671B

FROM vllm/vllm-openai:latest

# Enable HF Hub Transfer
ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Expose port 80
EXPOSE 80

# Entrypoint with API key
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            # name of the model
           "--model", "deepseek-ai/DeepSeek-R1", \
           # set the data type to bfloat16 - requires ~140GB GPU memory
           "--dtype", "bfloat16", \
           "--trust-remote-code", "true", \
           # below runs the model on 8 GPUs
           "--tensor-parallel-size","8", \
           # Maximum number of tokens, can lead to OOM if overestimated
           "--max-model-len", "4096", \
           # Port on which to run the vLLM server
           "--port", "80", \
           # API key for authentication to the server stored in Tensorfuse secrets
           "--api-key", "${VLLM_API_KEY}"]
```

We’ve configured the vLLM server with numerous CLI flags tailored to our specific use case. [A comprehensive list](https://docs.vllm.ai/en/v0.4.0.post1/serving/openai_compatible_server.html#command-line-arguments-for-the-server) of all
other vLLM flags is available for further reference, and if you have questions about selecting flags for production, the [Tensorfuse Community](https://join.slack.com/t/tensorfusecommunity/shared_invite/zt-2v64vkq51-VcToWhe5O~f9RppviZWPlg) is an excellent place to seek guidance.

### Step 3: Deployment config

Although you can deploy tensorfuse apps [using command line](/reference/cli_reference/tensorkube_deploy), it is always recommended to have a config file so
that you can follow a [GitOps approach](https://about.gitlab.com/topics/gitops/) to deployment.

```yaml deployment.yaml
# deployment.yaml for Deepseek-R1-671B

gpus: 8
gpu_type: h100
secret:
  - vllm-token
min-scale: 1
readiness:
  httpGet:
    path: /health
    port: 80
```

Don't forget the `readiness` endpoint in your config. Tensorfuse uses this endpoint to ensure that your service is healthy.


Now you can deploy your service using the following command:

```bash
tensorkube deploy --config-file ./deployment.yaml
```

### Step 4: Accessing the deployed app
Voila! Your **autoscaling** production LLM service is ready. Only authenticated requests will be served by your endpoint.

Once the deployment is successful, you can see the status of your app by running:

```bash
tensorkube deployment list
```

And that's it! You have successfully deployed the **world's strongest Open Source Reasoning Model**

To test it out, replace <YOUR_APP_URL> with the endpoint shown in the output of the above command and run:

```bash
curl --request POST \
  --url <YOUR_APP_URL>/v1/completions \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer vllm-key' \
  --data '{
    "model": "deepseek-ai/DeepSeek-R1",
    "prompt": "Earth to Robotland. What's up?",
    "max_tokens": 200
}'
```

Because vllm is compatible with the OpenAI API, you can use[OpenAI’s client libraries](https://platform.openai.com/docs/api-reference/completions/create)
as well. Here’s a sample snippet using Python:

```python
import openai

# Replace with your actual URL and token
base_url = "<YOUR_APP_URL>/v1"
api_key = "vllm-key"

openai.api_base = base_url
openai.api_key = api_key

response = openai.Completion.create(
    model="deepseek-ai/DeepSeek-R1",
    prompt="Hello, Deepseek R1! How are you today?",
    max_tokens=200
)

print(response)
```


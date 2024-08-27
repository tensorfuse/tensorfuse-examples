<p align="center">
  <a href="https://tensorfuse.io/">
    <img src="assets/Logo_whitebg.png" alt="Logo" width="200"/>
  </a>
</p>


![License](https://img.shields.io/badge/License-MIT-blue.svg)

# tensorfuse-examples
List of popular open-source models deployed on AWS using Tensorfuse. 

You can run them directly on GPU instances or deploy using the `tensorkube` runtime. The model will be served as an API that will auto-scale wrt the traffic you get. 

# Usage

Examples are organized into folders based on the modality. Each folder contains the FAST API code for model inference, and its environment as a Dockerfile. You can deploy them in two ways:

### Directly run on an EC2 instance

1. Launch and connect to an EC2 instance with Deep Learning AMI (it has Nvidia drivers installed).
2. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): This is important to provide GPU access to your containers
3. Build and deploy the Docker Container
   

### Run on serverless GPUs using Tensorfuse

1. You need to have aws-cli installed and configured on your local machine with admin access and `us-east-1` as the default region. [Follow these steps](https://docs.tensorfuse.io/guides/aws_cli) to configure. 
2. You need to have the quotas for the GPUs you are running. [Read about GPU quotas in AWS](https://tensorfuse.io/blog/increase-gpu-quota-on-aws-with-python-script) and how to apply for them.
3. Install the tensorfuse python package by running `pip install tensorkube`. Then, configure the tensorkube K8S cluster on your cloud by running, `tensorkube configure`. [More details here](https://docs.tensorfuse.io/getting_started_tensorkube)

To deploy the model, run the following command from the root directory of the model files:


```python
tensorkube deploy --gpus 1 --gpu-type a10g
```

Access the endpoint via:

```python
tensorkube list deployments
```

**Note:** If you encounter issues during deployment, refer to the [detailed instructions](https://docs.tensorfuse.io/introduction) for that model in our documentation.

# License
MIT License

# AWS Workshop Guide

Welcome to the Tensorfuse x AWS Workshop! This guide will walk you through all the steps required to provision infrastructure, configure access, and fine-tune models using Tensorfuse in your AWS account.

---

## 🟢 Initial Setup on Tensorfuse

1. **Log into Tensorfuse** - [Tensorfuse](https://app.tensorfuse.io)
2. **Configure the Cluster**
3. **Use AWS Identity Center**
   - Best practice for managing access securely.
4. **Grant Control Plane Access**
   - This sets up permissions for Tensorfuse to create and manage resources in your AWS account.
5. **Acknowledge Resource Creation**
   - Scroll down and **tick the checkbox** allowing CloudFormation to create resources on your behalf.
6. **Click “Create Stack”**
7. **Wait for Permission Check**
   - This may take up to 15 minutes (usually completes in ~3 minutes).
8. **Select Cluster Region**
   - We recommend `us-east-1` for high availability and lowest latency.
9. **Enter Alert Emails**
   - Provide the email addresses where you'd like to receive alerts for deployments and cluster status.

---

## 🐍 Local Environment Setup (Post Cluster Creation)

1. Ensure your Python version is `<= 3.11`. If needed, [download here](https://www.python.org/downloads/).
2. Create a virtual environment:  
   `python3 -m venv .venv`
3. Activate it:  
   `source .venv/bin/activate`
4. Clone the examples repo:  
   `git clone https://github.com/tensorfuse/tensorfuse-examples.git`
5. Move into the example folder:  
   `cd tensorfuse-examples/functioncalling_finetuning`
6. Install Python dependencies:  
   `pip install -r requirements.txt`
7. Check installed version:  
   `pip show tensorkube`
8. Install Tensorfuse prerequisites:  
   `tensorkube install-prerequisites`

---

## 🔐 Configure AWS Access (Identity Center Recommended)

9. Configure AWS profile:  
   `aws sso configure --profile tensorfuse`
10. Find your SSO URL in the IAM Identity Center dashboard.
11. Note the SSO region listed there as well.
12. Choose the **correct AWS account**.
13. Choose the **correct permission set**.
14. Set the region to where you created the cluster.
15. Set output format to `json`.
16. Log in to your profile:  
   `aws sso login --profile tensorfuse`

---

## 🧑‍💻 If Not Using Identity Center (Not Recommended)

17. Follow this fallback method:
   1. Go to your AWS account → Click your account name → **Security Credentials**
   2. Click **Create New Credentials**
   3. Copy your **Access Key ID** and **Secret Access Key**
   4. Run: `aws configure --profile tensorfuse`
   5. Enter copied keys when prompted
   6. Set region: `us-east-1`
   7. Run: `export AWS_PROFILE=tensorfuse`
   8. Sync cluster: `tensorkube sync`

---

## 🔑 Final Preparations

18. Login to Tensorfuse:  
   `tensorkube login`
19. Sync resources again:  
   `tensorkube sync`
20. Sanity check version:  
   `tensorkube version`

---

## 🤗 Hugging Face & Secrets Setup

21. Create Hugging Face token:  
   [Generate here](https://huggingface.co/settings/tokens)
22. Store it in Tensorfuse:  
   `tensorkube secret create hugging-face-secret HUGGING_FACE_HUB_TOKEN=hf_**`
23. Deploy the pipeline:  
   `tensorkube deploy --config-file ./deployment.yaml`

---

## 🧠 Finetuning Setup & Observability

24. Tensorfuse will now build the image and deploy the full fine-tuning + inference pipeline.
25. We’ll explore the internals of this deployment during the workshop.
26. Finetuning jobs are isolated using a separate environment called `keda`.

### Create Required Secrets in `keda`

27. Hugging Face secret for keda:  
   `tensorkube secret create hugging-face-secret HUGGING_FACE_HUB_TOKEN=hf_XX --env keda`
28. Optional: Add Weights & Biases secret for tracking training:  
   `tensorkube secret create wb-secret WANDB_API_KEY=7cXXX --env keda`

---

## 🛠 Run Finetuning Job

29. Finetuning involves two steps:  
   - **Base Job** → Defines training environment and resource specs.  
   - **Job Queue** → Queues actual training runs with different hyperparams.
30. Move to correct folder (`functioncalling_finetuning`) to pick the right Dockerfile.

### Deploy Base Job

31. Example command:  
   ```bash
   tensorkube job deploy --name qwen-finetuning --gpus 1 --gpu-type l40s --secret hugging-face-secret --secret wb-secret
   ```

### Queue a Training Run

32. Submit your run with:
   ```bash
   tensorkube job queue --job-name qwen-finetuning --job-id run-one --payload '{"hub_model_id":"<YOUR_HF_ACCOUNT>/qwen-functioncalling","num_epochs":1,"learning_rate":0.0002,"wandb_project":"qwen-functioncalling","wandb_entity":"<YOUR_WANDB_ENTITY_HERE>"}'
   ```

33. Monitor progress via [Weights & Biases](https://wandb.ai/home) under the project name `qwen-functioncalling`.

---

## ✅ Evaluate the Trained Model

34. Once the model is uploaded to Hugging Face, start evaluation:  
   `cd evals`  
   `python evaluation_script.py`

35. Choose `1` to **Load LoRA Adapter** and paste your Hugging Face model ID (e.g. `samagra-tensorfuse/qwen-functioncalling`)

36. Then run:  
   `python evaluation_script.py`  
   Choose `2` to **Run Evaluations** → Choose `3` for **All Models**

---

## 🔚 Cleanup

37. Delete deployment:  
   `tensorkube deployment delete inference-gpus-1-l40s`
38. In the AWS console, verify that only 2 `m5` instances remain.
39. Tear down the cluster:  
   `tensorkube teardown --remote`

---

Happy Finetuning! 🚀

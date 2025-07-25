```
tensorkube job deploy --name qwen-ft-w --gpus 1 --gpu-type l40s --secret hugging-face-secret --secret wb-secret
```

```
tensorkube secret create hugging-face-secret HUGGING_FACE_HUB_TOKEN=hf_na***** --env keda
```

```
tensorkube secret create wb-secret WANDB_API_KEY=7c***** --env keda
```


Jobs 

```
tensorkube job queue \
  --job-name qwen-ft-w \
  --job-id 3 \
  --payload '{"hub_model_id": "samagra-tensorfuse/qwen-ft-parsed", "num_epochs": 5, "learning_rate": 0.0001}'
```
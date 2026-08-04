# Amber LLM

From the original project https://github.com/LLM360

Après avoir téléchargé les données et les travailler

la suite est

```
.\run.ps1 -NodeCount 1 -RunWandb:$false -ExtraArgs @('--n_devices_per_node','1','--per_device_batch_size','2','--accumulate_grad_batches','8','--accelerator','cuda','--precision','bf16-mixed','--data_dir','../amber-data-prep/redpajama_tiny/merged/train')
```

Au depart on part sur `huggyllama/llama-7b`

ça fait OUT OF MEMEORY donc faut essayer avec des modèles moins puissants

```
Set-Location "C:\Users\nboud\Documents\GitHub\amber-llm\amber-train"
& "C:\Users\nboud\Documents\GitHub\amber-llm\.venv\Scripts\python.exe" main.py `
  --n_nodes 1 `
  --n_devices_per_node 1 `
  --per_device_batch_size 1 `
  --accumulate_grad_batches 16 `
  --accelerator cuda `
  --precision "bf16-mixed" `
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --data_dir ../amber-data-prep/redpajama_tiny/merged/train `
  --run_wandb False
```

ou encore

`HuggingFaceTB/SmolLM2-360M-Instruct`

sur ma RTX 3070 ça marche toutjours pas

et si je repasse sur un accelerator cpu ça bloque on dirait

il faut donc partir sur runpod.io mais ne le faites pas y a que les experts qui le font

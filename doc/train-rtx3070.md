# Entraînement local sur RTX 3070

Guide pas à pas pour lancer un entraînement **amber-pico** sur une machine locale équipée d'une RTX 3070 (8 Go VRAM) en utilisant le dataset **RedPajama Tiny**.

---

## Prérequis

- Python 3.10+
- CUDA 11.8 ou 12.1 installé
- ~20 Go d'espace disque libre

---

## Étape 1 — Installer les dépendances

```bash
cd amber-train
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install lightning>=2.1.2 transformers>=4.36.2 fire pytz tqdm numpy wandb
pip install flash-attn --no-build-isolation
```

> Si `flash-attn` échoue à la compilation, l'entraînement peut quand même fonctionner sans (performances légèrement réduites).

> **Si ta version CUDA système est 11.8**, remplace `cu121` par `cu118` dans la commande ci-dessus.

---

## Étape 2 — Télécharger les données tiny

```bash
cd amber-data-prep/redpajama_tiny
python download.py --out_root datasets
```

Télécharge les fichiers `*_sample.jsonl` depuis HuggingFace (`severo/RedPajama-Tiny`).

---

## Étape 3 — Tokeniser les données

```bash
python refine.py \
  --input_root datasets \
  --out_root refined \
  --tokenizer huggyllama/llama-7b \
  --concat_tokens 2049
```

Le tokenizer `huggyllama/llama-7b` sera téléchargé automatiquement depuis HuggingFace (~500 Mo).  
Chaque source (`arxiv`, `wikipedia`, etc.) sera tokenisée et sauvegardée dans `refined/<source>/train.jsonl`.

---

## Étape 4 — Merger et splitter les chunks

```bash
python tiny_mix_and_split.py \
  --input_root refined \
  --output_root merged \
  --num_split 1
```

Génère `merged/train/train_0.jsonl`, le fichier attendu par `main.py`.

---

## Étape 5 — Lancer l'entraînement

```bash
cd ../../amber-train

python main.py \
  --n_nodes 1 \
  --n_devices_per_node 1 \
  --per_device_batch_size 1 \
  --accumulate_grad_batches 16 \
  --accelerator cuda \
  --precision "bf16-mixed" \
  --model_name_or_path huggyllama/llama-160m \
  --data_dir ../amber-data-prep/redpajama_tiny/merged/train \
  --run_wandb False
```

> Une RTX 3070 8 Go ne peut pas entraîner `huggyllama/llama-7b` dans ce pipeline: le modèle OOM avant le premier step.
> Pour une validation locale réaliste, utilisez `huggyllama/llama-160m` avec un micro-batch de 1.
> `accumulate_grad_batches 16` simule un batch effectif de 16.

---

## Structure des dossiers attendue

```
amber-data-prep/
  redpajama_tiny/
    datasets/          ← téléchargé par download.py
    refined/           ← généré par refine.py
    merged/
      train/
        train_0.jsonl  ← utilisé par main.py ✅
```

---

## Dépannage — CUDA non détecté

Si le script indique qu'il n'y a pas d'accélérateur CUDA disponible, voici comment diagnostiquer le problème.

### 1. Vérifier que le GPU est visible

```bash
nvidia-smi
```

Si la commande échoue, le driver NVIDIA n'est pas actif → réinstaller le driver ou redémarrer la machine.

### 2. Vérifier que PyTorch détecte CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Si le résultat est `False`, PyTorch a été installé sans support CUDA.

### 3. Réinstaller PyTorch avec la bonne version CUDA

Vérifie ta version CUDA système avec `nvcc --version` ou `nvidia-smi`, puis installe la version correspondante :

```bash
# Pour CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Pour CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

> Un simple `pip install torch` sans `--index-url` installe la version CPU uniquement.

---

## Notes

- Le modèle 7B par défaut sert aux runs multi-GPU plus gros; il n'est pas réaliste sur une RTX 3070 8 Go.
- Pour un run local, `huggyllama/llama-160m` garde la même famille de tokenizer/config et permet de valider le pipeline sans OOM immédiat.
- Les checkpoints sont sauvegardés dans `amber-train/workdir_amber-pico/`.
- Pour reprendre un entraînement interrompu, relancer simplement la même commande : le script détecte automatiquement le dernier checkpoint.

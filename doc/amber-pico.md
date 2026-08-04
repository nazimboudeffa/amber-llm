# Amber Pico

Pipeline complet d'entraînement d'un petit modèle de langue à partir du dataset
[RedPajama Tiny](https://huggingface.co/datasets/severo/RedPajama-Tiny), dans le
même esprit que le projet Amber 7B (LLM360).

Le but est de valider le pipeline data-prep + training sur un petit jeu de données
avant de le passer à l'échelle.

## Vue d'ensemble

```
┌───────────────────── amber-data-prep/redpajama_tiny ─────────────────────┐
│                                                                          │
│  download.py  ──►  datasets/            (jsonl bruts par source)         │
│  refine.py    ──►  refined/<source>/    (tokenisé, 2049 tokens/seq)      │
│  tiny_mix_and_split.py ──► merged/train/train_<i>.jsonl + merged/valid/  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   amber-train/main.py   (train amber-pico)
```

## 1. Téléchargement

```powershell
cd amber-data-prep/redpajama_tiny
python download.py            # --> datasets/ (parquet + 7 fichiers *_sample.jsonl)
```

Sans dépendance externe (stdlib uniquement).

## 2. Tokenisation / concaténation

```powershell
python refine.py
```

- Tokenise chaque document avec le tokenizer LLaMA (`huggyllama/llama-7b` par défaut).
- Concatène les tokens avec `</s>` comme séparateur en séquences de 2049 tokens.
- Produit `refined/<source>/train.jsonl`, un fichier par source :
  `arxiv, book, c4, cc_2023-06, github, stackexchange, wikipedia`.
- Format compatible avec le reste du pipeline :
  `{"token_ids": [660, 29901, ...], "source": "arxiv"}`.

Options utiles : `--tokenizer`, `--concat_tokens 2049`, `--bos_text`, `--eos_text`,
`--truncate_num_samples N` (test rapide).

Dépendances : `transformers`, `tqdm` (voir `amber-data-prep/requirements.txt`).

## 3. Mix et split train/valid

```powershell
python tiny_mix_and_split.py --num_split 1 --num_valid_samples_per_subfolder 10
```

- Défauts : `--input_root refined`, `--output_root merged`,
  `--subfolders arxiv,book,c4,cc_2023-06,github,stackexchange,wikipedia`.
- Distribue les échantillons en round-robin dans `merged/train/train_<i>.jsonl`
  (naming attendu par `amber-train/main.py`).
- Sélectionne aléatoirement `num_valid_samples_per_subfolder` échantillons par
  source dans `merged/valid/<source>.jsonl`.
- Le shuffle n'est pas fait ici (fait au chargement dans le train).

## 4. Entraînement

```powershell
cd amber-train
python main.py --model_name_or_path huggyllama/llama-160m
```

`main.py` s'adapte automatiquement :

| Paramètre          | Défaut (auto)                                        |
| ------------------ | ---------------------------------------------------- |
| `data_dir`         | `amber-data-prep/redpajama_tiny/merged/train`        |
| `n_chunks`         | nb de fichiers `train_*.jsonl`                       |
| `examples_per_chunk` | nb de lignes de `train_0.jsonl`                    |
| `warmup_grad_steps`  | 10 % du nombre total de steps                      |
| `accelerator`      | `cuda` si dispo, sinon `cpu`                          |
| `precision`        | `bf16-mixed` (cuda) / `32-true` (cpu)                |
| `strategy`         | FSDP (cuda) / aucun (cpu)                             |
| `model_name_or_path` | `huggyllama/llama-7b`                              |

### CPU (validation locale)

Le modèle 7B nécessite ~28 Go de RAM (fp32) + l'optimiseur : inutilisable sur une
machine de dev. Pour tester le pipeline, utilisez un petit Llama de même
architecture :

```powershell
python main.py --model_name_or_path huggyllama/llama-160m
```

### GPU (run complet)

```powershell
python main.py --model_name_or_path huggyllama/llama-7b --n_devices_per_node 4
```

### Paramètres du run

- `MODEL_NAME = 'amber-pico'` : nom du run WandB et dossier de checkpoints
  (`workdir_amber-pico/ckpt_<i>/`).
- `per_device_batch_size`, `accumulate_grad_batches`, `n_nodes`,
  `n_devices_per_node`, `run_wandb`.

Les checkpoints sont sauvés dans `workdir_amber-pico/ckpt_<i>/` et le training
reprend automatiquement au dernier checkpoint via `get_last_ckpt_idx`.

## Structure des dossiers

```
amber-data-prep/
├── requirements.txt
└── redpajama_tiny/
    ├── download.py            # télécharge datasets/
    ├── refine.py              # tokenise -> refined/
    ├── tiny_mix_and_split.py  # merge/split -> merged/
    ├── datasets/              # données brutes (gitignoré)
    ├── refined/               # tokenisé par source (gitignoré)
    └── merged/                # chunks train + valid (gitignoré)

amber-train/
├── main.py                    # training amber-pico
├── main_utils.py              # chargement jsonl, lr schedule, checkpoints
├── model_utils/modeling_llama.py
└── requirements.txt
```

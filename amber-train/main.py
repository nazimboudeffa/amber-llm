from datetime import datetime
from pytz import timezone
import time
from functools import partial
import wandb
import glob
import os
import fire
import tqdm
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from transformers import AutoConfig, AutoTokenizer

from model_utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

from main_utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    get_last_ckpt_idx)


TIMEZONE = timezone('EST')
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]
MODEL_NAME = 'amber-pico'
PROJECT_NAME = 'amber'
RUN_NAME = f'pretraining_{MODEL_NAME}_{DATE}'
HF_MODEL_NAME_OR_PATH = 'huggyllama/llama-7b'
WORKDIR = f'workdir_{MODEL_NAME}'

LEARNING_RATE = 3e-4
LR_SCHEDULE_TYPE = 'cosine'
END_LEARNING_RATE = 3e-5
WARMUP_GRAD_STEPS = 2000
GRAD_NORM_CLIP = 1.
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
RANDOM_SEED = 11111
TRAIN_DATA_DIR = './data'
TRAIN_EXAMPLES_PER_CHUNK = 1706976
N_CHUNKS = 360

TINY_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'amber-data-prep', 'redpajama_tiny', 'merged', 'train')


def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def train_chunk(fabric,
                tokenizer,
                model,
                optimizer,
                lr_schedule_fn,
                examples,
                per_device_batch_size,
                accumulate_grad_batches,
                chunk_idx,
                run_wandb):
    step = chunk_idx * (len(examples) // per_device_batch_size)

    example_batch_idxes = tqdm.trange(
        0, len(examples), per_device_batch_size,
        desc=f'Training chunk {chunk_idx} (global_micro_batch_size='
             f'{per_device_batch_size * fabric.world_size}, '
             f'accumulate_grad_batches={accumulate_grad_batches})')
    for i in example_batch_idxes:
        t0 = time.time()

        lr = lr_schedule_fn(step)
        step += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (step % accumulate_grad_batches != 0)

        batch = collate_fn(
            examples=examples[i:i+per_device_batch_size], device=fabric.device)
        input_ids, labels = batch['input_ids'], batch['labels']
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape((-1, logits.size(-1))), labels.reshape(-1))

            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': loss.item(),
            'learning_rate': lr,
            'step': step,
            'speed(#tok/s/gpu)': int(input_ids.numel() / (time.time() - t0))
        }
        if not is_accumulating:
            log['grad_norm'] = grad_norm

        example_batch_idxes.set_postfix(log)
        if run_wandb and fabric.global_rank == 0:
            wandb.log(log)

    save_checkpoint(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{WORKDIR}/ckpt_{chunk_idx}')


def main(n_nodes=1,
         n_devices_per_node=4,
         per_device_batch_size=10,
         accumulate_grad_batches=1,
         run_wandb=False,
         data_dir=TINY_DATA_DIR,
         n_chunks=None,
         examples_per_chunk=None,
         warmup_grad_steps=None,
         accelerator=None,
         precision=None,
         model_name_or_path=HF_MODEL_NAME_OR_PATH):
    if n_chunks is None:
        n_chunks = len(glob.glob(f'{data_dir}/train_*.jsonl'))
    if examples_per_chunk is None:
        with open(f'{data_dir}/train_0.jsonl') as fin:
            examples_per_chunk = sum(1 for _ in fin)

    if accelerator is None:
        accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    if precision is None:
        precision = 'bf16-mixed' if accelerator == 'cuda' else '32-true'

    strategy = None
    if accelerator == 'cuda':
        strategy = FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}),
            activation_checkpointing_policy={LlamaDecoderLayer},
            cpu_offload=True,
            limit_all_gathers=True)

    devices = n_devices_per_node if accelerator == 'cuda' else 1

    fabric_kwargs = dict(
        accelerator=accelerator,
        num_nodes=n_nodes if accelerator == 'cuda' else 1,
        devices=devices,
        precision=precision)
    if strategy is not None:
        fabric_kwargs['strategy'] = strategy

    fabric = L.Fabric(**fabric_kwargs)
    fabric.launch()

    if fabric.global_rank == 0:
        os.makedirs(WORKDIR, exist_ok=True)
        if run_wandb:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    last_ckpt_idx = get_last_ckpt_idx(workdir=WORKDIR)
    fabric.seed_everything(RANDOM_SEED + last_ckpt_idx + 1)
    print(f'[amber-pico] accelerator={accelerator}, devices={devices}, '
          f'precision={precision}, strategy={type(strategy).__name__ if strategy else None}', flush=True)
    print(f'[amber-pico] loading model from {model_name_or_path} ...', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM(
        config=AutoConfig.from_pretrained(model_name_or_path))
    num_params = sum(p.numel() for p in model.parameters())
    print(f'[amber-pico] model ready: {num_params / 1e6:.1f}M params', flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False)

    model, optimizer = fabric.setup(model, optimizer)
    if last_ckpt_idx != -1:
        fabric.load(
            path=f'{WORKDIR}/ckpt_{last_ckpt_idx}/fabric_ckpt',
            state={'model': model, 'optimizer': optimizer})

    torch.cuda.empty_cache()

    global_micro_batch_size = per_device_batch_size * fabric.world_size
    total_steps = examples_per_chunk // global_micro_batch_size * n_chunks
    if warmup_grad_steps is None:
        warmup_grad_steps = max(1, total_steps // 10)
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=warmup_grad_steps * accumulate_grad_batches,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)
    print(f'[amber-pico] data_dir={data_dir}, n_chunks={n_chunks}, '
          f'examples_per_chunk={examples_per_chunk}, total_steps={total_steps}', flush=True)

    for chunk_idx in range(last_ckpt_idx + 1, n_chunks):
        print(f'[amber-pico] loading chunk {chunk_idx}/{n_chunks} ...', flush=True)
        examples = load_jsonl_examples(
            filename=f'{data_dir}/train_{chunk_idx}.jsonl',
            n_examples=examples_per_chunk,
            shuffle=True,
            global_micro_batch_size=global_micro_batch_size,
            global_rank=fabric.global_rank,
            world_size=fabric.world_size)

        train_chunk(
            fabric=fabric,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            chunk_idx=chunk_idx,
            run_wandb=run_wandb)


if __name__ == '__main__':
    fire.Fire(main)
"""
Baseline de image captioning sobre VizWiz-Captions: fine-tuning de BLIP-base con LoRA.

Diseñado para correr en WSL2 con una GPU de 8 GB (RTX 5060). Usa el loader limpio
`wizviz_captions.py`. Entrena sobre pares (imagen, caption) y evalúa en val con
CIDEr / BLEU-4 / ROUGE-L (todo vía pycocoevalcap, sin dependencia de Java).

Ejemplos:
    # Smoke test rápido (valida el pipeline entero en unos minutos)
    python baseline_blip_lora.py --max-train-samples 500 --max-eval-samples 200 --epochs 1

    # Entrenamiento completo
    python baseline_blip_lora.py --epochs 3 --batch-size 8 --grad-accum 2

    # Solo evaluar el modelo base sin fine-tuning (línea de referencia)
    python baseline_blip_lora.py --no-train --max-eval-samples 500
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# El loader vive junto a este script (raíz del repo).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions  # noqa: E402


# --------------------------------------------------------------------------- #
# Dataset / collator
# --------------------------------------------------------------------------- #
class CaptionDataset(Dataset):
    """Devuelve (imagen PIL, caption) desde el dataset explotado del loader."""

    def __init__(self, hf_ds):
        self.paths = hf_ds["image_path"]
        self.caps = hf_ds["caption"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {"image": img, "caption": self.caps[i]}


def make_collate(processor):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        images = [b["image"] for b in batch]
        texts = [b["caption"] for b in batch]
        enc = processor(
            images=images, text=texts, padding=True,
            truncation=True, max_length=40, return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[labels == pad_id] = -100  # ignorar padding en la loss
        enc["labels"] = labels
        return enc

    return collate


# --------------------------------------------------------------------------- #
# Métricas (pycocoevalcap, tokenización propia -> sin Java)
# --------------------------------------------------------------------------- #
_punct = re.compile(r"[^a-z0-9 ]+")


def _tok(s: str) -> str:
    return _punct.sub(" ", s.lower()).strip()


def caption_metrics(gts: dict[int, list[str]], res: dict[int, list[str]]) -> dict:
    """gts: id->refs, res: id->[hyp]. Devuelve CIDEr, BLEU-1..4, ROUGE-L."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts_t = {k: [_tok(x) for x in v] for k, v in gts.items()}
    res_t = {k: [_tok(v[0])] for k, v in res.items()}

    out = {}
    bleu, bleu_list = Bleu(4).compute_score(gts_t, res_t)
    for i, b in enumerate(bleu):
        out[f"BLEU-{i+1}"] = round(100 * b, 2)
    out["ROUGE-L"] = round(100 * Rouge().compute_score(gts_t, res_t)[0], 2)
    out["CIDEr"] = round(100 * Cider().compute_score(gts_t, res_t)[0], 2)
    return out


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, processor, val_grouped, device, args) -> dict:
    model.eval()
    ids = val_grouped["image_id"]
    paths = val_grouped["image_path"]
    refs = val_grouped["captions"]

    n = min(args.max_eval_samples, len(ids)) if args.max_eval_samples else len(ids)
    gts, res = {}, {}
    bs = args.eval_batch_size
    t0 = time.time()
    for start in range(0, n, bs):
        sl = slice(start, min(start + bs, n))
        imgs = [Image.open(p).convert("RGB") for p in paths[sl]]
        pix = processor(images=imgs, return_tensors="pt").pixel_values.to(device, model.dtype)
        gen = model.generate(pixel_values=pix, max_new_tokens=args.max_new_tokens,
                             num_beams=args.num_beams)
        caps = processor.batch_decode(gen, skip_special_tokens=True)
        for j, idx in enumerate(range(start, min(start + bs, n))):
            gts[ids[idx]] = refs[idx]
            res[ids[idx]] = [caps[j].strip()]
        print(f"  eval {min(start+bs, n)}/{n}", end="\r", flush=True)
    dt = time.time() - t0
    print(f"\n  generación: {dt:.1f}s ({n/dt:.1f} img/s)")

    m = caption_metrics(gts, res)
    # muestra un par de ejemplos
    sample_ids = list(res)[:3]
    for sid in sample_ids:
        print(f"  [{sid}] pred: {res[sid][0]!r}")
        print(f"        ref : {gts[sid][0]!r}")
    return m


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--output-dir", default=str(Path.home() / "wizviz" / "runs" / "blip_lora"))
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    args = p.parse_args()
    if args.max_train_samples == 0:
        args.max_train_samples = None
    if args.max_eval_samples == 0:
        args.max_eval_samples = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | modelo: {args.model}")

    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(args.model, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # ---- datos ----
    print("cargando dataset (limpio)...")
    train_ds = load_wizviz_captions(
        root=args.data_root, splits=("train",), format="exploded", with_images=False
    )["train"]
    val_grouped = load_wizviz_captions(
        root=args.data_root, splits=("val",), format="grouped", with_images=False
    )["val"]
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    print(f"train: {len(train_ds)} pares | val: {len(val_grouped)} imágenes")

    # ---- LoRA ----
    if not args.no_train:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            target_modules=["query", "value"],  # atención del decoder de texto de BLIP
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable()

        from transformers import Trainer, TrainingArguments

        targs = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            fp16=(device == "cuda"),
            logging_steps=20,
            save_strategy="no",
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to=[],
        )
        trainer = Trainer(
            model=model, args=targs,
            train_dataset=CaptionDataset(train_ds),
            data_collator=make_collate(processor),
        )
        print("=== entrenando ===")
        trainer.train()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        print(f"adaptador LoRA guardado en {args.output_dir}")

    # ---- evaluación ----
    print("=== evaluando en val ===")
    metrics = evaluate(model, processor, val_grouped, device, args)
    print("\n=== MÉTRICAS (val) ===")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v}")


if __name__ == "__main__":
    main()

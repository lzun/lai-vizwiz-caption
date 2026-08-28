"""
Prototipo: fine-tuning de GIT-large (microsoft/git-large-coco) con LoRA sobre
VizWiz-Captions, usando el mismo pipeline que el baseline de BLIP (loader limpio,
LoRA texto+visión, eval CIDEr/BLEU/ROUGE por subconjunto). Para comparar contra
el mejor BLIP (texto+visión: 64.73 en 20k, 74.60 en full).

GIT es un image-to-text causal LM: concatena embeddings de imagen + texto y hace
LM causal sobre el texto. Módulos de atención para LoRA:
  texto  = query, value (transformer GIT)
  visión = q_proj, v_proj (CLIP ViT-L)

Ejemplos:
    python baseline_git.py --max-train-samples 20000 --epochs 4 --batch-size 8
    python baseline_git.py --no-train   # evaluar GIT-large-coco sin fine-tuning
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions      # noqa: E402
from baseline_blip_lora import caption_metrics         # noqa: E402

MAX_CAP = 40


class CaptionDataset(Dataset):
    def __init__(self, hf_ds):
        self.paths = hf_ds["image_path"]
        self.caps = hf_ds["caption"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": Image.open(self.paths[i]).convert("RGB"), "caption": self.caps[i]}


def make_collate(processor):
    pad = processor.tokenizer.pad_token_id

    def collate(batch):
        enc = processor(images=[b["image"] for b in batch], text=[b["caption"] for b in batch],
                        padding=True, truncation=True, max_length=MAX_CAP, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels == pad] = -100
        enc["labels"] = labels
        return enc

    return collate


@torch.no_grad()
def evaluate(model, processor, val, device, args):
    model.eval()
    n = min(args.max_eval_samples, len(val)) if args.max_eval_samples else len(val)
    gts, res, is_text = {}, {}, {}
    bs = args.eval_batch_size
    t0 = time.time()
    for s in range(0, n, bs):
        idx = list(range(s, min(s + bs, n)))
        imgs = [Image.open(val["image_path"][k]).convert("RGB") for k in idx]
        pix = processor(images=imgs, return_tensors="pt").pixel_values.to(device, model.dtype)
        gen = model.generate(pixel_values=pix, num_beams=args.num_beams,
                             max_length=args.max_new_tokens)
        caps = processor.batch_decode(gen, skip_special_tokens=True)
        for k, c in zip(idx, caps):
            gts[val["image_id"][k]] = val["captions"][k]
            res[val["image_id"][k]] = [c.strip()]
            is_text[val["image_id"][k]] = bool(val["text_detected"][k])
        if (s + bs) % 400 < bs:
            print(f"  eval {min(s+bs, n)}/{n} ({min(s+bs, n)/(time.time()-t0):.1f} img/s)", flush=True)

    def sub(pred):
        g = {i: gts[i] for i in gts if pred(i)}
        return len(g), (caption_metrics(g, {i: res[i] for i in g}) if g else {})
    print("\n=== ejemplos ===")
    for i in list(res)[:3]:
        print(f"  pred: {res[i][0]!r}\n  ref : {gts[i][0]!r}")
    return [("all", (len(gts), caption_metrics(gts, res))),
            ("text", sub(lambda i: is_text[i])),
            ("notext", sub(lambda i: not is_text[i]))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--model", default="microsoft/git-large-coco")
    p.add_argument("--output-dir", default=str(Path.home() / "wizviz" / "runs" / "git_large"))
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="query,value,q_proj,v_proj")
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--max-eval-samples", type=int, default=2000)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--save-steps", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | modelo: {args.model}")

    from transformers import AutoProcessor, GitForCausalLM
    processor = AutoProcessor.from_pretrained(args.model)
    model = GitForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    print("cargando dataset...")
    train = load_wizviz_captions(root=args.data_root, splits=("train",),
                                 format="exploded", with_images=False)["train"]
    val = load_wizviz_captions(root=args.data_root, splits=("val",),
                               format="grouped", with_images=False)["val"]
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    print(f"train: {len(train)} pares | val: {len(val)} imágenes")

    if not args.no_train:
        from peft import LoraConfig, get_peft_model
        targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", target_modules=targets))
        model.print_trainable_parameters()
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        from transformers import Trainer, TrainingArguments
        save_kw = ({"save_strategy": "steps", "save_steps": args.save_steps, "save_total_limit": 2}
                   if args.save_steps > 0 else {"save_strategy": "no"})
        targs = TrainingArguments(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
            num_train_epochs=args.epochs, fp16=(device == "cuda"), logging_steps=25,
            remove_unused_columns=False, dataloader_num_workers=2, report_to=[], **save_kw)
        trainer = Trainer(model=model, args=targs, train_dataset=CaptionDataset(train),
                          data_collator=make_collate(processor))
        from transformers.trainer_utils import get_last_checkpoint
        last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
        print(f"=== entrenando (GIT-large){' [reanuda]' if last else ''} ===")
        trainer.train(resume_from_checkpoint=last)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)

    print("=== evaluando en val ===")
    for name, (cnt, m) in evaluate(model, processor, val, device, args):
        print(f"  [{name:6s} n={cnt:4d}] " + " ".join(f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()

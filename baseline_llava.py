"""
Prototipo: QLoRA (4-bit) de LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) sobre VizWiz-Captions,
con el mismo loader/métricas. LLaVA es English-only (Vicuna) → sin fuga de idioma.

En 8 GB: 4-bit + batch 1 + grad-accum + grad-checkpointing. LLaVA-1.5 usa CLIP-336
(576 tokens de imagen fijos, no reducibles). Formato chat; se entrena SOLO la respuesta
del asistente (el resto -> -100).

Uso:
    python baseline_llava.py --max-train-samples 20000 --epochs 2 --save-steps 200
    python baseline_llava.py --no-train --max-eval-samples 100   # zero-shot
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
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics

MODEL = "llava-hf/llava-1.5-7b-hf"
PROMPT = "Provide one short caption describing this image, in the style of a brief crowdsourced description."
ASST = "ASSISTANT:"


class CaptionDataset(Dataset):
    def __init__(self, hf_ds):
        self.paths, self.caps = hf_ds["image_path"], hf_ds["caption"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": self.paths[i], "caption": self.caps[i]}


def build_text(processor, caption=None):
    content = [{"type": "image"}, {"type": "text", "text": PROMPT}]
    conv = [{"role": "user", "content": content}]
    if caption is not None:
        conv.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
    return processor.apply_chat_template(conv, add_generation_prompt=(caption is None))


def make_collate(processor, image_token_id):
    tok = processor.tokenizer
    asst_ids = tok(ASST, add_special_tokens=False).input_ids

    def collate(batch):
        imgs = [Image.open(b["image"]).convert("RGB") for b in batch]
        texts = [build_text(processor, b["caption"]) for b in batch]
        enc = processor(images=imgs, text=texts, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        for r in range(labels.size(0)):
            ids = enc["input_ids"][r].tolist()
            start = None
            for j in range(len(ids) - len(asst_ids), -1, -1):
                if ids[j:j + len(asst_ids)] == asst_ids:
                    start = j + len(asst_ids); break
            if start is not None:
                labels[r, :start] = -100
        enc["labels"] = labels
        return enc

    return collate


@torch.no_grad()
def evaluate(model, processor, val, device, args):
    model.eval()
    tok = processor.tokenizer
    n = min(args.max_eval_samples, len(val)) if args.max_eval_samples else len(val)
    gts, res, is_text = {}, {}, {}
    t0 = time.time()
    for s in range(0, n, args.eval_batch_size):
        idx = list(range(s, min(s + args.eval_batch_size, n)))
        imgs = [Image.open(val["image_path"][k]).convert("RGB") for k in idx]
        texts = [build_text(processor, None) for _ in idx]
        enc = processor(images=imgs, text=texts, padding=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
        trimmed = [o[enc["input_ids"].shape[1]:] for o in gen]
        caps = tok.batch_decode(trimmed, skip_special_tokens=True)
        for k, c in zip(idx, caps):
            gts[val["image_id"][k]] = val["captions"][k]
            res[val["image_id"][k]] = [c.strip()]
            is_text[val["image_id"][k]] = bool(val["text_detected"][k])
        if (s + args.eval_batch_size) % 200 < args.eval_batch_size:
            print(f"  eval {min(s+args.eval_batch_size,n)}/{n} ({min(s+args.eval_batch_size,n)/(time.time()-t0):.2f} img/s)", flush=True)

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
    p.add_argument("--model", default=MODEL)
    p.add_argument("--output-dir", default=str(Path.home() / "wizviz" / "runs" / "llava15_7b"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--max-eval-samples", type=int, default=2000)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--save-steps", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
    print(f"device: {device} | modelo: {args.model}")

    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    image_token_id = model.config.image_token_index

    train = load_wizviz_captions(root=args.data_root, splits=("train",), format="exploded",
                                 with_images=False)["train"]
    val = load_wizviz_captions(root=args.data_root, splits=("val",), format="grouped",
                               with_images=False)["val"]
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    print(f"train: {len(train)} pares | val: {len(val)} imágenes")

    if not args.no_train:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

        from transformers import Trainer, TrainingArguments
        save_kw = ({"save_strategy": "steps", "save_steps": args.save_steps, "save_total_limit": 2}
                   if args.save_steps > 0 else {"save_strategy": "no"})
        targs = TrainingArguments(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
            num_train_epochs=args.epochs, fp16=True, logging_steps=25, remove_unused_columns=False,
            dataloader_num_workers=2, report_to=[], gradient_checkpointing=True, **save_kw)
        trainer = Trainer(model=model, args=targs, train_dataset=CaptionDataset(train),
                          data_collator=make_collate(processor, image_token_id))
        from transformers.trainer_utils import get_last_checkpoint
        last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
        print(f"=== entrenando (LLaVA-1.5-7B QLoRA){' [reanuda]' if last else ''} ===")
        trainer.train(resume_from_checkpoint=last)
        model.save_pretrained(args.output_dir)

    print("=== evaluando en val ===")
    for name, (cnt, m) in evaluate(model, processor, val, device, args):
        print(f"  [{name:6s} n={cnt:4d}] " + " ".join(f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()

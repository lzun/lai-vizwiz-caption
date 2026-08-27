"""
Fase 2 — Captioning con OCR: fine-tuning de BLIP-base + LoRA inyectando el texto OCR
(EasyOCR precomputado por `ocr_precompute.py`) como PREFIJO condicional del decoder.

Comparación controlada A/B sobre el MISMO subset:
    * control  (--ocr off / omitir --ocr): captioning normal, como el baseline.
    * tratamiento (--ocr): el decoder ve  [CLS] + <tokens OCR> + <caption>  y solo se
      entrena la parte de la caption (el prefijo OCR se enmascara en la loss).

En inferencia, con OCR se genera a partir de [CLS]+<OCR> y se recorta el prefijo por
longitud de tokens. Se reportan métricas globales y desglosadas por `text_detected`.

Ejemplos:
    python baseline_blip_ocr.py --ocr-cache ocr_cache_easyocr.json --ocr \
        --max-train-samples 20000 --max-eval-samples 2000 --epochs 2 --batch-size 16
    python baseline_blip_ocr.py --max-train-samples 20000 --max-eval-samples 2000 \
        --epochs 2 --batch-size 16      # control sin OCR
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions          # noqa: E402
from baseline_blip_lora import caption_metrics             # noqa: E402

MAX_OCR = 32   # tokens máximos del prefijo OCR
MAX_CAP = 40   # tokens máximos de la caption


def clean_ocr(text: str, mode: str) -> str:
    """Limpia el texto OCR según la fuente.
    - none: EasyOCR crudo (sin cambios).
    - ade_full: salida de Landing AI ADE sin tags (descripción + texto transcrito).
    - ade_textonly: ADE quitando los bloques <description> (solo texto transcrito).
    """
    if not text or mode == "none":
        return text or ""
    t = re.sub(r"<!--.*?-->", " ", text, flags=re.S)          # quita comentarios doc_id
    if mode == "ade_textonly":
        t = re.sub(r"<description>.*?</description>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)                            # quita tags restantes
    return re.sub(r"\s+", " ", t).strip()


class CaptionOCRDataset(Dataset):
    def __init__(self, paths, captions, ocrs):
        self.paths, self.caps, self.ocrs = paths, captions, ocrs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": Image.open(self.paths[i]).convert("RGB"),
                "caption": self.caps[i], "ocr": self.ocrs[i]}


def make_collate(processor, use_ocr):
    tok = processor.tokenizer
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

    def collate(batch):
        pixel_values = processor(images=[b["image"] for b in batch],
                                 return_tensors="pt").pixel_values
        seqs, labels = [], []
        for b in batch:
            cap_ids = tok(b["caption"], add_special_tokens=False,
                          truncation=True, max_length=MAX_CAP).input_ids
            ocr_ids = []
            if use_ocr and b["ocr"].strip():
                ocr_ids = tok(b["ocr"], add_special_tokens=False,
                              truncation=True, max_length=MAX_OCR).input_ids
            ids = [cls] + ocr_ids + cap_ids + [sep]
            lab = list(ids)
            for j in range(1 + len(ocr_ids)):   # enmascara CLS + prefijo OCR
                lab[j] = -100
            seqs.append(ids)
            labels.append(lab)
        maxlen = max(len(s) for s in seqs)
        input_ids, attn, lab_pad = [], [], []
        for s, l in zip(seqs, labels):
            n = maxlen - len(s)
            input_ids.append(s + [pad] * n)
            attn.append([1] * len(s) + [0] * n)
            lab_pad.append(l + [-100] * n)
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(lab_pad),
        }

    return collate


@torch.no_grad()
def evaluate(model, processor, val, ocr_cache, use_ocr, device, args):
    model.eval()
    tok = processor.tokenizer
    cls = tok.cls_token_id
    n = min(args.max_eval_samples, len(val)) if args.max_eval_samples else len(val)

    gts, res, is_text = {}, {}, {}
    t0 = time.time()
    for k in range(n):
        img = Image.open(val["image_path"][k]).convert("RGB")
        pix = processor(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
        ocr = clean_ocr(ocr_cache.get(val["file_name"][k], ""), args.ocr_clean) if use_ocr else ""
        if ocr.strip():
            ocr_ids = tok(ocr, add_special_tokens=False, truncation=True,
                          max_length=MAX_OCR).input_ids
            prompt = torch.tensor([[cls] + ocr_ids], device=device)
            gen = model.generate(pixel_values=pix, input_ids=prompt,
                                 num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                                 min_new_tokens=args.min_new_tokens)
            cont = gen[0][1 + len(ocr_ids):]          # quita [DEC] + prefijo OCR
            cap = tok.decode(cont, skip_special_tokens=True).strip()
        else:
            gen = model.generate(pixel_values=pix, num_beams=args.num_beams,
                                 max_new_tokens=args.max_new_tokens,
                                 min_new_tokens=args.min_new_tokens)
            cap = tok.decode(gen[0], skip_special_tokens=True).strip()
        iid = val["image_id"][k]
        gts[iid] = val["captions"][k]
        res[iid] = [cap]
        is_text[iid] = bool(val["text_detected"][k])
        if (k + 1) % 250 == 0 or k + 1 == n:
            print(f"  eval {k+1}/{n} ({(k+1)/(time.time()-t0):.1f} img/s)", flush=True)

    def subset(pred):
        g = {i: gts[i] for i in gts if pred(i)}
        r = {i: res[i] for i in g}
        return caption_metrics(g, r) if g else {}

    print("\n=== ejemplos ===")
    for iid in list(res)[:3]:
        print(f"  pred: {res[iid][0]!r}\n  ref : {gts[iid][0]!r}")

    return {
        "all":    (len(gts),                       caption_metrics(gts, res)),
        "text":   (sum(is_text.values()),          subset(lambda i: is_text[i])),
        "notext": (sum(1 - v for v in is_text.values()), subset(lambda i: not is_text[i])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--ocr-cache", default="ocr_cache_easyocr.json")
    p.add_argument("--ocr", action="store_true", help="usar el prefijo OCR (si no, es el control)")
    p.add_argument("--ocr-clean", default="none", choices=["none", "ade_full", "ade_textonly"],
                   help="limpieza del texto OCR: none=EasyOCR, ade_*=Landing AI")
    p.add_argument("--output-dir", default="")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="query,value",
                   help="módulos LoRA (p. ej. 'query,value,qkv' para texto+visión)")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--save-steps", type=int, default=0, help="si >0, checkpoint cada N pasos (reanudable)")
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--min-new-tokens", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--max-eval-samples", type=int, default=2000)
    p.add_argument("--no-train", action="store_true")
    args = p.parse_args()
    tag = "ocr" if args.ocr else "control"
    if not args.output_dir:
        args.output_dir = str(Path.home() / "wizviz" / "runs" / f"blip_{tag}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | modo: {tag.upper()}")

    ocr_cache = {}
    if args.ocr:
        ocr_cache = json.loads(Path(args.ocr_cache).read_text(encoding="utf-8"))
        with_txt = sum(1 for v in ocr_cache.values() if v.strip())
        print(f"cache OCR: {len(ocr_cache)} imgs, {with_txt} con texto")

    from transformers import BlipForConditionalGeneration, BlipProcessor
    processor = BlipProcessor.from_pretrained(args.model, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    print("cargando dataset...")
    train = load_wizviz_captions(root=args.data_root, splits=("train",),
                                 format="exploded", with_images=False)["train"]
    val = load_wizviz_captions(root=args.data_root, splits=("val",),
                               format="grouped", with_images=False)["val"]
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    train_ocrs = [clean_ocr(ocr_cache.get(fn, ""), args.ocr_clean) if args.ocr else ""
                  for fn in train["file_name"]]
    print(f"train: {len(train)} pares | val: {len(val)} imágenes")

    if not args.no_train:
        from peft import LoraConfig, get_peft_model
        targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", target_modules=targets))  # texto (query/value) y opc. visión (qkv)
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
        trainer = Trainer(
            model=model, args=targs,
            train_dataset=CaptionOCRDataset(train["image_path"], train["caption"], train_ocrs),
            data_collator=make_collate(processor, use_ocr=args.ocr))
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
        print(f"=== entrenando ({tag}){' [reanuda '+last_ckpt+']' if last_ckpt else ''} ===")
        trainer.train(resume_from_checkpoint=last_ckpt)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        print(f"adaptador guardado en {args.output_dir}")

    print(f"=== evaluando ({tag}) en val ===")
    results = evaluate(model, processor, val, ocr_cache, args.ocr, device, args)
    print(f"\n=== MÉTRICAS ({tag.upper()}) ===")
    for name, (cnt, m) in results.items():
        line = " ".join(f"{k}={v}" for k, v in m.items())
        print(f"  [{name:6s} n={cnt:4d}] {line}")


if __name__ == "__main__":
    main()

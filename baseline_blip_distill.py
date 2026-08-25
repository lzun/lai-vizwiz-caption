"""
Fase 4 — Destilación desde Landing AI ADE (sin dependencia en inferencia).

La Fase 3 mostró que inyectar la descripción de ADE como PREFIJO mejora el CIDEr
(+3.2 en imágenes con texto), pero exige llamar a Landing AI también en inferencia.
Aquí se prueba **destilar** ese conocimiento: se usa la descripción de ADE como una
caption pseudo-objetivo EXTRA en entrenamiento (no como prefijo), para que BLIP la
interiorice y en inferencia genere solo, gratis y autónomo.

Experimento A/B vs el control (mismo subset de 20k pares humanos):
    * control:   solo captions humanas (= baseline_blip_ocr sin --ocr).
    * destilado: captions humanas + 1 pseudo-caption de ADE por imagen con texto.

La pseudo-caption es la PRIMERA ORACIÓN de la descripción de ADE (caption-like),
no las ~93 palabras completas, para no arrastrar el estilo verboso.

Uso:
    python baseline_blip_distill.py --ade-cache ocr_cache_landingai.json \
        --max-train-samples 20000 --max-eval-samples 2000 --epochs 2 --batch-size 16
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
from baseline_blip_ocr import clean_ocr                    # noqa: E402

MAX_CAP = 40


def ade_pseudo_caption(md: str) -> str:
    """Primera oración DESCRIPTIVA de ADE, limpia y caption-like.

    Solo acepta descripciones de escena (empiezan por 'A/An/The/This/Close/Image/Photo'
    y terminan en punto), descartando las salidas que son transcripción cruda de texto
    (etiquetas en mayúsculas, barras de estado con emojis, etc.), que como target dañarían.
    """
    t = clean_ocr(md, "ade_full")
    if not t:
        return ""
    t = re.sub(r"[^\x00-\x7f]+", " ", t)          # fuera emojis / no-ascii
    t = re.sub(r"\s+", " ", t).strip()
    m = re.match(r"(.{0,220}?[.!?])(\s|$)", t)     # primera oración con puntuación terminal
    if not m:
        return ""
    s = m.group(1).strip()
    words = s.split()
    if not (4 <= len(words) <= 30):
        return ""
    if not re.match(r"(A|An|The|This|Close|Image|Photo)\b", s):
        return ""
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.4:
        return ""                                   # descarta transcripciones en mayúsculas
    return s


class CaptionDS(Dataset):
    def __init__(self, paths, caps):
        self.paths, self.caps = paths, caps

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
                             max_new_tokens=args.max_new_tokens)
        caps = processor.batch_decode(gen, skip_special_tokens=True)
        for k, c in zip(idx, caps):
            gts[val["image_id"][k]] = val["captions"][k]
            res[val["image_id"][k]] = [c.strip()]
            is_text[val["image_id"][k]] = bool(val["text_detected"][k])
    print(f"  eval {n} imgs ({n/(time.time()-t0):.1f} img/s)")

    def sub(pred):
        g = {i: gts[i] for i in gts if pred(i)}
        return len(g), (caption_metrics(g, {i: res[i] for i in g}) if g else {})
    return [("all", (len(gts), caption_metrics(gts, res))),
            ("text", sub(lambda i: is_text[i])),
            ("notext", sub(lambda i: not is_text[i]))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--ade-cache", default="ocr_cache_landingai.json")
    p.add_argument("--distill", action="store_true", help="añadir pseudo-captions de ADE (si no, control)")
    p.add_argument("--teacher-cache", default="", help="Fase 4b: usar captions del profesor (json) como target en vez de la prosa de ADE")
    p.add_argument("--output-dir", default="")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--max-eval-samples", type=int, default=2000)
    args = p.parse_args()
    tag = "distill" if args.distill else "control"
    if not args.output_dir:
        args.output_dir = str(Path.home() / "wizviz" / "runs" / f"blip_{tag}2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | modo: {tag.upper()}")

    from transformers import BlipForConditionalGeneration, BlipProcessor
    processor = BlipProcessor.from_pretrained(args.model, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    train = load_wizviz_captions(root=args.data_root, splits=("train",),
                                 format="exploded", with_images=False)["train"].select(
                                     range(args.max_train_samples))
    val = load_wizviz_captions(root=args.data_root, splits=("val",),
                               format="grouped", with_images=False)["val"]

    paths = list(train["image_path"])
    caps = list(train["caption"])
    n_human = len(caps)
    n_pseudo = 0
    if args.distill:
        teacher = json.loads(Path(args.teacher_cache).read_text(encoding="utf-8")) if args.teacher_cache else None
        ade = json.loads(Path(args.ade_cache).read_text(encoding="utf-8"))
        seen = set()
        for fn, pth in zip(train["file_name"], train["image_path"]):
            if fn in seen:
                continue
            seen.add(fn)
            # Fase 4b: target del profesor (estilo-VizWiz). Fase 4: prosa cruda de ADE.
            ps = teacher.get(fn, "") if teacher is not None else ade_pseudo_caption(ade.get(fn, ""))
            if ps and len(ps.split()) >= 3:
                paths.append(pth); caps.append(ps); n_pseudo += 1
    print(f"pares: {n_human} humanos + {n_pseudo} pseudo-ADE = {len(caps)} | val: {len(val)}")

    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             bias="none", target_modules=["query", "value"]))
    model.print_trainable_parameters()

    from transformers import Trainer, TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, num_train_epochs=args.epochs, fp16=(device == "cuda"),
        logging_steps=25, save_strategy="no", remove_unused_columns=False,
        dataloader_num_workers=2, report_to=[])
    Trainer(model=model, args=targs, train_dataset=CaptionDS(paths, caps),
            data_collator=make_collate(processor)).train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)

    print(f"=== evaluando ({tag}) en val ===")
    for name, (cnt, m) in evaluate(model, processor, val, device, args):
        print(f"  [{name:6s} n={cnt:4d}] " + " ".join(f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()

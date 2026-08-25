"""
Fase 4b — Genera los targets del profesor para destilación estilo-matched.

El profesor es el modelo de la Fase 3 (`blip_ade_full`), que produce captions cortas
estilo-VizWiz cuando ve imagen + prefijo ADE. Se corre CON el prefijo ADE sobre las
mismas imágenes que augmentó la Fase 4, y sus captions se guardan como targets de
destilación (teacher_captions.json). Luego el alumno se entrena con ellas SIN prefijo.

Uso:
    python distill_teacher_gen.py --adapter ~/wizviz/runs/blip_ade_full \
        --ade-cache ocr_cache_landingai.json --out teacher_captions.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions          # noqa: E402
from baseline_blip_ocr import clean_ocr, MAX_OCR          # noqa: E402
from baseline_blip_distill import ade_pseudo_caption      # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--adapter", default=str(Path.home() / "wizviz" / "runs" / "blip_ade_full"))
    p.add_argument("--ade-cache", default="ocr_cache_landingai.json")
    p.add_argument("--out", default="teacher_captions.json")
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--min-new-tokens", type=int, default=5)
    args = p.parse_args()
    root = args.data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ade = json.loads(Path(args.ade_cache).read_text(encoding="utf-8"))
    tr = load_wizviz_captions(root=root, splits=("train",), format="exploded",
                              with_images=False)["train"].select(range(args.max_train_samples))
    # mismas imágenes que augmentó la Fase 4: donde ade_pseudo_caption era válida
    targets, seen = [], set()
    for fn, pth in zip(tr["file_name"], tr["image_path"]):
        if fn in seen:
            continue
        seen.add(fn)
        if ade_pseudo_caption(ade.get(fn, "")):
            targets.append((fn, pth))
    print(f"imágenes objetivo (profesor): {len(targets)}")

    from transformers import BlipForConditionalGeneration, BlipProcessor
    from peft import PeftModel
    processor = BlipProcessor.from_pretrained(args.model, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    tok = processor.tokenizer
    cls = tok.cls_token_id

    out = Path(args.out)
    cache = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    t0 = time.time()
    for i, (fn, pth) in enumerate(targets):
        if fn in cache:
            continue
        img = Image.open(pth).convert("RGB")
        pix = processor(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
        ocr_ids = tok(clean_ocr(ade[fn], "ade_full"), add_special_tokens=False,
                      truncation=True, max_length=MAX_OCR).input_ids
        prompt = torch.tensor([[cls] + ocr_ids], device=device)
        gen = model.generate(pixel_values=pix, input_ids=prompt, num_beams=args.num_beams,
                             max_new_tokens=args.max_new_tokens, min_new_tokens=args.min_new_tokens)
        cap = tok.decode(gen[0][1 + len(ocr_ids):], skip_special_tokens=True).strip()
        cache[fn] = cap
        if (i + 1) % 200 == 0 or i + 1 == len(targets):
            out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"  {i+1}/{len(targets)} | {(i+1)/(time.time()-t0):.1f} img/s", flush=True)
    out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"listo. {len(cache)} captions del profesor -> {out}")
    for fn in list(cache)[:4]:
        print(f"  {fn}: {cache[fn]!r}")


if __name__ == "__main__":
    main()

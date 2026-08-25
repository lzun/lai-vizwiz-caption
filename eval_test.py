"""
Self-evaluation en el TEST set de VizWiz-Captions.

============================ ⚠️  ADVERTENCIA  ⚠️ ============================
Este script NO produce una evaluación válida. Las captions de test de VizWiz son
oficialmente OCULTAS. Se intentó reconstruir el GT desde el CSV con el mapeo
`archivo i ↔ CSV VizWiz_test_(20000+i)`, pero ese mapeo está ROTO: alinea en los
índices 0 y 100 pero falla en índices tardíos (5000, 7000 → imagen y captions no
corresponden). El resultado (CIDEr ~5-6) es un artefacto de desalineación.
Usa `val` como held-out (ver baseline_blip_lora.py). Se conserva solo como referencia.
============================================================================

El ground-truth "está" en `VizWiz_captions_project_all_metadata.csv` con nombres
ORIGINALES, pero no hay forma fiable de mapearlos a los archivos renombrados.

Construye el GT (lo cachea en test_gt.json), evalúa el modelo indicado y reporta
CIDEr/BLEU/ROUGE globales y desglosados por `text_detected`.

Uso:
    python eval_test.py                                  # mejor modelo (LoRA full)
    python eval_test.py --base-only                      # BLIP base sin fine-tuning
    python eval_test.py --adapter ~/wizviz/runs/blip_lora --max-samples 2000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions      # noqa: E402
from baseline_blip_lora import caption_metrics         # noqa: E402

DEFAULT_ADAPTER = str(Path.home() / "wizviz" / "runs" / "blip_lora_full")
CSV = "VizWiz_captions_project_all_metadata.csv"


def build_test_gt(root: Path, caption_col="IMG_CAPTION_SPELLCHECKED", cache="test_gt.json"):
    """Reconstruye {file_name: {...}} del test desde el CSV. Cachea a JSON."""
    out = root / cache
    if out.exists():
        print(f"GT de test cacheado: {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    import pandas as pd
    test = load_wizviz_captions(root=str(root), splits=("test",),
                                format="grouped", with_images=False)["test"]
    m = pd.read_csv(root / CSV, usecols=["IMG", caption_col, "TRAIN_VAL_TEST"], encoding="latin-1")
    m = m[m["TRAIN_VAL_TEST"] == "TEST"]
    byimg = defaultdict(list)
    for img, cap in zip(m["IMG"], m[caption_col]):
        if isinstance(cap, str) and cap.strip():
            byimg[img].append(cap.strip())

    gt = {}
    missing = 0
    for k in range(len(test)):
        fn = test["file_name"][k]
        i = int(re.search(r"_(\d+)\.jpg", fn).group(1))
        caps = byimg.get(f"VizWiz_test_{20000 + i:012d}.jpg", [])
        if caps:
            gt[fn] = {"image_path": test["image_path"][k],
                      "text_detected": bool(test["text_detected"][k]),
                      "captions": caps}
        else:
            missing += 1
    out.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    print(f"GT de test construido: {len(gt)} imágenes ({missing} sin captions) -> {out}")
    return gt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    args = p.parse_args()
    root = Path(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = build_test_gt(root)
    files = sorted(gt)
    if args.max_samples:
        files = files[: args.max_samples]
    print(f"evaluando {len(files)} imágenes de test | device {device}")

    from transformers import BlipForConditionalGeneration, BlipProcessor
    processor = BlipProcessor.from_pretrained(args.model, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    tag = "base"
    if not args.base_only and Path(args.adapter).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = Path(args.adapter).name
        print(f"adaptador: {args.adapter}")
    model.eval()

    gts, res, is_text = {}, {}, {}
    bs = args.batch_size
    t0 = time.time()
    for s in range(0, len(files), bs):
        chunk = files[s:s + bs]
        imgs = [Image.open(gt[f]["image_path"]).convert("RGB") for f in chunk]
        pix = processor(images=imgs, return_tensors="pt").pixel_values.to(device, model.dtype)
        gen = model.generate(pixel_values=pix, num_beams=args.num_beams,
                             max_new_tokens=args.max_new_tokens)
        caps = processor.batch_decode(gen, skip_special_tokens=True)
        for f, c in zip(chunk, caps):
            gts[f] = gt[f]["captions"]
            res[f] = [c.strip()]
            is_text[f] = gt[f]["text_detected"]
        if (s + bs) % 800 < bs:
            print(f"  {min(s+bs, len(files))}/{len(files)} "
                  f"({min(s+bs, len(files))/(time.time()-t0):.1f} img/s)", flush=True)

    def sub(pred):
        g = {k: gts[k] for k in gts if pred(k)}
        return len(g), (caption_metrics(g, {k: res[k] for k in g}) if g else {})

    print("\n=== ejemplos ===")
    for f in files[:3]:
        print(f"  pred: {res[f][0]!r}\n  ref : {gts[f][0]!r}")

    print(f"\n=== TEST · modelo: {tag} · {len(gts)} imágenes ===")
    for name, (cnt, m) in [("all", (len(gts), caption_metrics(gts, res))),
                           ("text", sub(lambda k: is_text[k])),
                           ("notext", sub(lambda k: not is_text[k]))]:
        line = " ".join(f"{k}={v}" for k, v in m.items())
        print(f"  [{name:6s} n={cnt:4d}] {line}")


if __name__ == "__main__":
    main()

"""Re-evalúa un adapter BLIP+prefijo (OCR/ADE) ya entrenado, con la función evaluate()
corregida de baseline_blip_ocr (fix del bug input_ids[:, :-1] de BLIP.generate).
No re-entrena ni usa créditos: lee el caché OCR/ADE existente.

Ejemplo:
    python reeval_ocr.py --adapter ~/wizviz/runs/blip_ade_vision_full \
        --ocr-cache ocr_cache_landingai.json --ocr-clean ade_full
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_ocr import evaluate

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--ocr-cache", required=True)
p.add_argument("--ocr-clean", default="ade_full", choices=["none", "ade_full", "ade_textonly"])
p.add_argument("--max-eval-samples", type=int, default=2000)
p.add_argument("--num-beams", type=int, default=3)
p.add_argument("--max-new-tokens", type=int, default=30)
p.add_argument("--min-new-tokens", type=int, default=5)
a = p.parse_args()

root = str(Path(__file__).resolve().parent)
device = "cuda"
from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import PeftModel
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", dtype=torch.float16).to(device)
model = PeftModel.from_pretrained(model, str(Path(a.adapter).expanduser())).eval()

val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
cache = json.loads((Path(root) / a.ocr_cache).read_text(encoding="utf-8"))

args = SimpleNamespace(max_eval_samples=a.max_eval_samples, num_beams=a.num_beams,
                       max_new_tokens=a.max_new_tokens, min_new_tokens=a.min_new_tokens,
                       ocr_clean=a.ocr_clean)
print(f"adapter={Path(a.adapter).name} cache={a.ocr_cache} clean={a.ocr_clean}")
results = evaluate(model, proc, val, cache, use_ocr=True, device=device, args=args)
print("=== MÉTRICAS (corregido) ===")
for name, (cnt, m) in results.items():
    print(f"  [{name:6s} n={cnt:4d}] " + " ".join(f"{k}={v}" for k, v in m.items()))

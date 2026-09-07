"""Desglose BLEU-1..4 + ROUGE-L + CIDEr de LLaVA (completa y 1er enunciado)
a partir de las captions cacheadas en llava_caps.json. Solo CPU, sin regenerar."""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics
from eval_llava_extra import first_sentence

root = str(Path(__file__).resolve().parent)
val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
refs_by_id = {val["image_id"][i]: val["captions"][i] for i in range(len(val["image_id"]))}

data = json.loads((Path(root) / "llava_caps.json").read_text(encoding="utf-8"))
ids, caps = data["ids"], data["caps"]
gts = {i: refs_by_id[i] for i in ids}


def show(variant_caps, label):
    res = {ids[k]: [variant_caps[k]] for k in range(len(ids))}
    m = caption_metrics(gts, res)
    print(f"\n=== LLaVA · {label} ===")
    print(f"BLEU-1={m['BLEU-1']}  BLEU-2={m['BLEU-2']}  BLEU-3={m['BLEU-3']}  "
          f"BLEU-4={m['BLEU-4']}  ROUGE-L={m['ROUGE-L']}  CIDEr={m['CIDEr']}")


show(caps, "completa")
show([first_sentence(c) for c in caps], "primer enunciado")

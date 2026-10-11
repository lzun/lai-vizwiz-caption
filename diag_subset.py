"""Diagnóstico: compara solo-visión (blip_scaled4) vs visión+ADE (blip_ade_vision_full)
por subconjunto sobre EXACTAMENTE las mismas imágenes, para distinguir efecto del modelo
del efecto del conjunto de evaluación (sensibilidad de CIDEr al IDF del corpus).

Usa las captions cacheadas en fullval_caps/ :
  - _val_scaled4.json / _val_ade.json  -> primeras 2000
  - blip_scaled4.json  / blip_ade_vision_full.json -> val completo (7542)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics

root = Path(__file__).resolve().parent
CAPS = root / "fullval_caps"
val = load_wizviz_captions(root=str(root), splits=("val",), format="grouped", with_images=False)["val"]
refs = {str(val["image_id"][i]): val["captions"][i] for i in range(len(val["image_id"]))}
istext = {str(val["image_id"][i]): bool(val["text_detected"][i]) for i in range(len(val["image_id"]))}


def cider_subsets(caps_file):
    d = json.loads((CAPS / caps_file).read_text(encoding="utf-8"))["caps"]
    ids = list(d.keys())
    def c(keep):
        g = {i: refs[i] for i in ids if keep(i)}
        r = {i: [d[i]] for i in g}
        return len(g), (caption_metrics(g, r).get("CIDEr") if g else None)
    return {"all": c(lambda i: True), "text": c(lambda i: istext[i]),
            "notext": c(lambda i: not istext[i])}


for label, vis, ade in [("PRIMERAS 2000", "_val_scaled4.json", "_val_ade.json"),
                        ("VAL COMPLETO", "blip_scaled4.json", "blip_ade_vision_full.json")]:
    print(f"\n===== {label} =====")
    sv = cider_subsets(vis); sa = cider_subsets(ade)
    for sub in ("all", "text", "notext"):
        nv, cv = sv[sub]; na, ca = sa[sub]
        delta = (ca - cv) if (ca is not None and cv is not None) else None
        print(f"  {sub:6s} n={nv:5d} | solo-vision CIDEr={cv} | vision+ADE CIDEr={ca} | ADE-visión Δ={delta:+.2f}" if delta is not None else f"  {sub}: n/a")

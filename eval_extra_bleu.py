"""Recupera BLEU-4 + ROUGE-L (+ CIDEr sanity-check) de los 4 modelos de la comparativa
a partir de las captions cacheadas en metrics_extra_full.json. Solo CPU, sin regenerar."""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics

root = Path(__file__).resolve().parent
val = load_wizviz_captions(root=str(root), splits=("val",), format="grouped", with_images=False)["val"]
refs_by_file = {Path(val["image_path"][i]).name: val["captions"][i]
                for i in range(len(val["image_path"]))}

d = json.loads((root / "metrics_extra_full.json").read_text(encoding="utf-8"))
files, caps_by_model, saved = d["files"], d["captions"], d["results"]
gts = {k: refs_by_file[files[k]] for k in range(len(files))}

for model, caps in caps_by_model.items():
    res = {k: [caps[k]] for k in range(len(files))}
    m = caption_metrics(gts, res)
    ref_cider = saved[model]["CIDEr"]
    ok = "OK" if abs(m["CIDEr"] - ref_cider) < 0.5 else f"MISMATCH (guardado {ref_cider})"
    print(f"{model:32s} BLEU-4={m['BLEU-4']:5.2f}  ROUGE-L={m['ROUGE-L']:5.2f}  "
          f"CIDEr={m['CIDEr']:5.2f} [{ok}]")

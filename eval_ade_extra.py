"""Recomputa la fila BLIP+ADE de §7 (métricas complementarias) con la generación
CORREGIDA (fix del bug de prefijo de BLIP.generate). Genera captions de
blip_ade_vision_full + ADE sobre las mismas 2000 val y computa CIDEr + CLIP + BERT + SBERT.
Las guarda para no regenerar (ade_caps_fixed.json)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics
from eval_metrics_extra import clip_score, bert_score_f1, sbert_sim
import compare_models as cm

RUNS = Path.home() / "wizviz" / "runs"
root = str(Path(__file__).resolve().parent)
device = "cuda"

val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
n = min(2000, len(val))
paths = [val["image_path"][i] for i in range(n)]
files = [val["file_name"][i] for i in range(n)]
refs = [val["captions"][i] for i in range(n)]
ids = [val["image_id"][i] for i in range(n)]

ade = json.loads((Path(root) / "ocr_cache_landingai.json").read_text(encoding="utf-8"))
caps = cm.gen_blip(paths, files, RUNS / "blip_ade_vision_full", ade, device)
(Path(root) / "ade_caps_fixed.json").write_text(
    json.dumps({"ids": ids, "caps": caps}, ensure_ascii=False), encoding="utf-8")

import statistics
wlen = statistics.mean(len(c.split()) for c in caps)
gts = {ids[i]: refs[i] for i in range(n)}
res = {ids[i]: [caps[i]] for i in range(n)}
m = caption_metrics(gts, res)
cs = clip_score(paths, caps, device)
bs = bert_score_f1(caps, refs)
ss = sbert_sim(caps, refs, device)
print("\n=== BLIP + ADE (corregido, 2000 val) ===")
print(f"palabras/caption media: {wlen:.1f}")
print(f"CIDEr={m['CIDEr']}  BLEU-4={m['BLEU-4']}  ROUGE-L={m['ROUGE-L']}  "
      f"CLIPScore={cs:.2f}  BERTScore={100*bs:.2f}  SBERT={100*ss:.2f}")

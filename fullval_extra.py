"""Tabla 5 sobre val COMPLETO: CIDEr/BLEU-4/ROUGE-L + CLIPScore/BERTScore/SBERT para los
6 modelos de comparación, desde las captions cacheadas en fullval_caps/.
LLaVA (1er enunciado) se deriva truncando las captions de LLaVA."""
from __future__ import annotations
import json, re, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics
from eval_metrics_extra import clip_score, bert_score_f1, sbert_sim

root = Path(__file__).resolve().parent
CAPS = root / "fullval_caps"
device = "cuda"

val = load_wizviz_captions(root=str(root), splits=("val",), format="grouped", with_images=False)["val"]
ids = [str(val["image_id"][i]) for i in range(len(val["image_id"]))]
refs = {str(val["image_id"][i]): val["captions"][i] for i in range(len(ids))}
path = {str(val["image_id"][i]): val["image_path"][i] for i in range(len(ids))}


def first_sentence(t):
    t = t.strip()
    m = re.search(r"[.!?](\s|$)", t)
    return t[: m.end()].strip() if m else t


def load_caps(name):
    return json.loads((CAPS / f"{name}.json").read_text(encoding="utf-8"))["caps"]


llava = load_caps("llava")
models = {
    "BLIP + vision": load_caps("blip_scaled4"),
    "BLIP + ADE": load_caps("blip_ade_vision_full"),
    "GIT-large": load_caps("git"),
    "Qwen2.5-VL": load_caps("qwen"),
    "LLaVA (raw)": llava,
    "LLaVA (1st sentence)": {k: first_sentence(v) for k, v in llava.items()},
}

print(f"{'Model':24s} {'words':>6s} {'CIDEr':>7s} {'BLEU4':>6s} {'ROUGE':>6s} {'CLIP':>5s} {'BERT':>6s} {'SBERT':>6s}")
for name, caps in models.items():
    order = [i for i in ids if i in caps]
    capl = [caps[i] for i in order]
    paths = [path[i] for i in order]
    refl = [refs[i] for i in order]
    gts = {i: refs[i] for i in order}
    res = {i: [caps[i]] for i in order}
    m = caption_metrics(gts, res)
    wl = statistics.mean(len(c.split()) for c in capl)
    cs = clip_score(paths, capl, device)
    bs = 100 * bert_score_f1(capl, refl)
    ss = 100 * sbert_sim(capl, refl, device)
    print(f"{name:24s} {wl:6.1f} {m['CIDEr']:7.2f} {m['BLEU-4']:6.2f} {m['ROUGE-L']:6.2f} "
          f"{cs:5.2f} {bs:6.2f} {ss:6.2f}", flush=True)

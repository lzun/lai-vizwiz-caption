"""
Métricas complementarias a BLEU/CIDEr para evaluar mejor la utilidad de las captions
(especialmente ADE, que es fiel a la imagen aunque diverja de las refs humanas):

  * CLIPScore  — alineación imagen↔caption (REFERENCE-FREE): ¿la caption describe la imagen?
  * BERTScore  — similitud semántica caption↔refs (F1, multi-ref).
  * SBERT-sim  — similitud de oración caption↔refs (coseno máx, Sentence-BERT).

Genera captions de cada modelo sobre una muestra de val (cargando uno a la vez) y
reporta CIDEr + las 3 métricas nuevas, para una comparación más justa.

Uso:
    python eval_metrics_extra.py --num 300 --out metrics_extra.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics
import compare_models as cm

RUNS = Path.home() / "wizviz" / "runs"


# ---------- CLIPScore (reference-free) ----------
@torch.no_grad()
def clip_score(paths, caps, device, bs=16):
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    scores = []
    for s in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + bs]]
        txt = [c if c.strip() else " " for c in caps[s:s + bs]]
        inp = proc(text=txt, images=imgs, return_tensors="pt", padding=True,
                   truncation=True, max_length=77).to(device)
        out = model(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        cos = (ie * te).sum(-1).float()
        scores += (2.5 * cos.clamp(min=0)).tolist()  # escalado estándar CLIPScore
    del model, proc; cm.free()
    return sum(scores) / len(scores)


# ---------- BERTScore (semántica, multi-ref) ----------
def bert_score_f1(caps, refs):
    from bert_score import score
    P, R, F1 = score(caps, refs, lang="en", rescale_with_baseline=True, verbose=False)
    cm.free()
    return F1.mean().item()


# ---------- SBERT similarity (coseno máx sobre refs) ----------
@torch.no_grad()
def sbert_sim(caps, refs, device):
    from sentence_transformers import SentenceTransformer, util
    sb = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    cand = sb.encode(caps, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    sims = []
    for i, rs in enumerate(refs):
        re = sb.encode(rs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        sims.append(util.cos_sim(cand[i], re).max().item())
    del sb; cm.free()
    return sum(sims) / len(sims)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="metrics_extra.json")
    args = p.parse_args()
    root = str(Path(__file__).resolve().parent)
    device = "cuda"

    val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
    N = min(2000, len(val))
    idx = list(range(N)); random.seed(args.seed); random.shuffle(idx)
    if args.num:
        idx = idx[: args.num]
    paths = [val["image_path"][i] for i in idx]
    files = [val["file_name"][i] for i in idx]
    refs = [val["captions"][i] for i in idx]
    ids = [val["image_id"][i] for i in idx]
    print(f"{len(idx)} imágenes de muestra")

    ade = json.loads((Path(root) / "ocr_cache_landingai.json").read_text(encoding="utf-8"))

    plan = [
        ("BLIP + visión (74.6)", lambda: cm.gen_blip(paths, files, RUNS / "blip_scaled4", None, device)),
        ("BLIP + visión + ADE (78.7)", lambda: cm.gen_blip(paths, files, RUNS / "blip_ade_vision_full", ade, device)),
        ("GIT-large (73.9)", lambda: cm.gen_git(paths, RUNS / "git_coco_full_r32", device)),
        ("Qwen2.5-VL", lambda: cm.gen_qwen(paths, RUNS / "qwen25vl_20k", device)),
    ]

    caps_by_model = {}
    for name, fn in plan:
        print(f">> generando captions: {name}", flush=True)
        caps_by_model[name] = fn()

    results = {}
    for name, caps in caps_by_model.items():
        print(f">> métricas: {name}", flush=True)
        gts = {ids[i]: refs[i] for i in range(len(idx))}
        res = {ids[i]: [caps[i]] for i in range(len(idx))}
        cider = caption_metrics(gts, res).get("CIDEr")
        cs = clip_score(paths, caps, device)
        bs = bert_score_f1(caps, refs)
        ss = sbert_sim(caps, refs, device)
        results[name] = {"CIDEr": cider, "CLIPScore": round(cs, 2),
                         "BERTScore": round(100 * bs, 2), "SBERT": round(100 * ss, 2)}
        print(f"   {results[name]}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"results": results, "captions": caps_by_model, "files": files}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n=== TABLA (muestra de %d val) ===" % len(idx))
    hdr = f"{'Modelo':30s} {'CIDEr':>7s} {'CLIPScore':>10s} {'BERTScore':>10s} {'SBERT':>7s}"
    print(hdr)
    for name, m in results.items():
        print(f"{name:30s} {m['CIDEr']:>7} {m['CLIPScore']:>10} {m['BERTScore']:>10} {m['SBERT']:>7}")


if __name__ == "__main__":
    main()

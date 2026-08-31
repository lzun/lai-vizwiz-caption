"""
Galería HTML comparando las captions de cada modelo del proyecto sobre las mismas
imágenes de val, junto a las referencias humanas. Carga un modelo a la vez (para
caber en 8 GB), genera para toda la muestra, libera memoria y pasa al siguiente.

Uso:
    python compare_models.py --num 10 --html comparacion_modelos.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_ocr import clean_ocr, MAX_OCR

RUNS = Path.home() / "wizviz" / "runs"
BLIP = "Salesforce/blip-image-captioning-base"


def free():
    import gc
    gc.collect(); torch.cuda.empty_cache()


def load_imgs(paths):
    return [Image.open(p).convert("RGB") for p in paths]


# ---- BLIP (base / +visión / +ADE) ----
def gen_blip(paths, files, adapter, ade_cache, device):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(BLIP, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(BLIP, dtype=torch.float16).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    tok = proc.tokenizer
    out = []
    for p, fn in zip(paths, files):
        img = Image.open(p).convert("RGB")
        pix = proc(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
        ocr = clean_ocr(ade_cache.get(fn, ""), "ade_full") if ade_cache is not None else ""
        with torch.no_grad():
            if ocr.strip():
                ocr_ids = tok(ocr, add_special_tokens=False, truncation=True, max_length=MAX_OCR).input_ids
                prompt = torch.tensor([[tok.cls_token_id] + ocr_ids], device=device)
                g = model.generate(pixel_values=pix, input_ids=prompt, num_beams=3,
                                   max_new_tokens=30, min_new_tokens=5)
                cap = tok.decode(g[0][1 + len(ocr_ids):], skip_special_tokens=True).strip()
            else:
                g = model.generate(pixel_values=pix, num_beams=3, max_new_tokens=30)
                cap = tok.decode(g[0], skip_special_tokens=True).strip()
        out.append(cap)
    del model, proc; free()
    return out


# ---- GIT-large ----
def gen_git(paths, adapter, device):
    from transformers import AutoProcessor, GitForCausalLM
    from peft import PeftModel
    name = "microsoft/git-large-coco"
    proc = AutoProcessor.from_pretrained(name)
    model = GitForCausalLM.from_pretrained(name, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        pix = proc(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
        with torch.no_grad():
            g = model.generate(pixel_values=pix, num_beams=3, max_length=40)
        out.append(proc.batch_decode(g, skip_special_tokens=True)[0].strip())
    del model, proc; free()
    return out


# ---- Qwen2.5-VL (4-bit + bloqueo CJK) ----
def gen_qwen(paths, adapter, device):
    import baseline_qwen as bq
    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig)
    from peft import PeftModel
    name = "Qwen/Qwen2.5-VL-3B-Instruct"
    bq.PROMPT = ("Describe this image in one short English caption, in the style of a brief "
                 "crowdsourced description. Always answer in English only.")
    proc = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=448 ** 2, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    tok = proc.tokenizer
    # ids CJK a suprimir
    def is_cjk(ch):
        o = ord(ch)
        return (0x3000 <= o <= 0x303F or 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)
    suppress = [i for t, i in tok.get_vocab().items() if any(is_cjk(c) for c in tok.convert_tokens_to_string([t]))]
    collate = bq.make_collate(proc, 448, train=False)
    out = []
    for p in paths:
        enc = collate([{"image": p, "caption": ""}]).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, num_beams=1, suppress_tokens=suppress)
        trimmed = g[0][enc["input_ids"].shape[1]:]
        out.append(tok.decode(trimmed, skip_special_tokens=True).strip())
    del model, proc; free()
    return out


def data_uri(path, max_side=420):
    im = Image.open(path).convert("RGB"); im.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def write_html(out_path, cards, model_names):
    css = """
    body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#0f141b;color:#e8ecf2}
    h1{font-family:Georgia,serif;font-size:23px;margin:0 0 4px}
    .sub{color:#95a1b3;font-size:14px;margin:0 0 24px}
    .card{display:grid;grid-template-columns:320px 1fr;gap:22px;background:#161d27;border:1px solid #26303c;
      border-radius:14px;padding:18px;margin-bottom:18px}
    .card img{width:100%;border-radius:8px;background:#000;display:block}
    .fn{font-family:ui-monospace,monospace;font-size:11.5px;color:#6b7788;margin-top:8px;word-break:break-all}
    .tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;margin-top:8px}
    .txt{background:#183330;color:#4fb0a8}.notxt{background:#26303c;color:#95a1b3}
    table{border-collapse:collapse;width:100%}
    td{padding:9px 10px;border-top:1px solid #26303c;font-size:14px;vertical-align:top}
    td.m{color:#95a1b3;white-space:nowrap;width:1%;font-weight:600;font-size:12.5px}
    tr.human td{color:#a78bfa}tr.best td.m{color:#e8a34a}tr.auto td.m{color:#7ee787}
    .cap{color:#e8ecf2}
    @media(max-width:720px){.card{grid-template-columns:1fr}}
    """
    P = [f"<!doctype html><meta charset='utf-8'><title>Comparación de modelos — VizWiz</title><style>{css}</style>",
         "<h1>Comparación de captions por modelo — VizWiz-Captions</h1>",
         f"<p class='sub'>{len(cards)} imágenes de val · cada modelo genera su caption · morado = referencias humanas</p>"]
    for c in cards:
        P.append("<div class='card'>")
        tag = "<span class='tag txt'>con texto</span>" if c["text"] else "<span class='tag notxt'>sin texto</span>"
        P.append(f"<div><img src='{c['uri']}'>{tag}<div class='fn'>{html.escape(c['file'])}</div></div>")
        P.append("<table>")
        for r in (c["refs"] or [])[:2]:
            P.append(f"<tr class='human'><td class='m'>humano</td><td class='cap'>{html.escape(r)}</td></tr>")
        for mn in model_names:
            cls = "best" if "ADE" in mn else ("auto" if "74.6" in mn else "")
            P.append(f"<tr class='{cls}'><td class='m'>{html.escape(mn)}</td><td class='cap'>{html.escape(c['caps'].get(mn,''))}</td></tr>")
        P.append("</table></div>")
    Path(out_path).write_text("\n".join(P), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--html", default="comparacion_modelos.html")
    p.add_argument("--text-only", action="store_true", help="solo imágenes con texto (resalta el OCR)")
    args = p.parse_args()
    root = str(Path(__file__).resolve().parent)
    device = "cuda"

    val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
    # muestra diversa dentro de las primeras 2000 (que tienen ADE cacheado)
    N = min(2000, len(val))
    txt = [i for i in range(N) if val["text_detected"][i]]
    notx = [i for i in range(N) if not val["text_detected"][i]]
    random.seed(args.seed); random.shuffle(txt); random.shuffle(notx)
    k = args.num
    idx = txt[:k] if args.text_only else txt[: (2 * k) // 3] + notx[: k - (2 * k) // 3]
    paths = [val["image_path"][i] for i in idx]
    files = [val["file_name"][i] for i in idx]
    refs = [val["captions"][i] for i in idx]
    is_text = [bool(val["text_detected"][i]) for i in idx]
    print(f"{len(idx)} imágenes de muestra")

    ade = json.loads((Path(root) / "ocr_cache_landingai.json").read_text(encoding="utf-8"))

    caps_by_model = {}
    plan = [
        ("BLIP base", lambda: gen_blip(paths, files, None, None, device)),
        ("BLIP + visión (74.6)", lambda: gen_blip(paths, files, RUNS / "blip_scaled4", None, device)),
        ("BLIP + visión + ADE (78.7)", lambda: gen_blip(paths, files, RUNS / "blip_ade_vision_full", ade, device)),
        ("GIT-large (73.9)", lambda: gen_git(paths, RUNS / "git_coco_full_r32", device)),
        ("Qwen2.5-VL", lambda: gen_qwen(paths, RUNS / "qwen25vl_20k", device)),
    ]
    for name, fn in plan:
        print(f">> generando: {name}")
        try:
            caps_by_model[name] = fn()
        except Exception as e:
            print(f"   [error {name}]: {str(e)[:150]}")
            caps_by_model[name] = ["[error]"] * len(paths)

    model_names = [n for n, _ in plan]
    cards = []
    for j, (pth, fl, rf, tx) in enumerate(zip(paths, files, refs, is_text)):
        cards.append({"uri": data_uri(pth), "file": fl, "refs": rf, "text": tx,
                      "caps": {mn: caps_by_model[mn][j] for mn in model_names}})
    write_html(args.html, cards, model_names)
    print(f"galería: {args.html}")


if __name__ == "__main__":
    main()

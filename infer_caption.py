"""
Inferencia / demo de captioning con el BLIP-base + LoRA fine-tuneado en VizWiz-Captions.

Carga el adaptador LoRA sobre BLIP-base y genera captions para imágenes sueltas,
una carpeta, o una muestra de un split del dataset. Puede comparar el modelo base
(sin fine-tuning) contra el fine-tuneado, y volcar una galería HTML autocontenida.

Ejemplos (dentro de WSL, con el venv ~/wizviz/.venv):
    # Imágenes concretas
    python infer_caption.py foto1.jpg foto2.jpg

    # 8 imágenes aleatorias del test set, comparando base vs fine-tuned, a HTML
    python infer_caption.py --split test --num 8 --random --compare --html demo.html

    # Muestra de val (incluye las referencias humanas en el HTML)
    python infer_caption.py --split val --num 12 --html demo_val.html

    # Modelo base sin adaptador
    python infer_caption.py --split test --num 4 --base-only
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions  # noqa: E402

DEFAULT_ADAPTER = str(Path.home() / "wizviz" / "runs" / "blip_lora_full")


# --------------------------------------------------------------------------- #
# Carga del modelo
# --------------------------------------------------------------------------- #
def load_model(model_name, adapter, base_only, device):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(model_name, dtype=dtype).to(device)

    has_adapter = False
    if not base_only:
        if not Path(adapter).exists():
            print(f"[aviso] adaptador no encontrado en {adapter} -> uso solo modelo base")
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            has_adapter = True
            print(f"adaptador LoRA cargado: {adapter}")
    model.eval()
    return processor, model, has_adapter


@torch.no_grad()
def caption(model, processor, images, device, num_beams, max_new_tokens):
    pix = processor(images=images, return_tensors="pt").pixel_values.to(device, model.dtype)
    gen = model.generate(pixel_values=pix, num_beams=num_beams, max_new_tokens=max_new_tokens)
    return [c.strip() for c in processor.batch_decode(gen, skip_special_tokens=True)]


# --------------------------------------------------------------------------- #
# Recolección de imágenes a captionar
# --------------------------------------------------------------------------- #
def gather_images(args):
    """Devuelve lista de (path, refs|None)."""
    root = str(Path(__file__).resolve().parent)
    if args.split:
        ds = load_wizviz_captions(root=root, splits=(args.split,),
                                  format="grouped", with_images=False)[args.split]
        idxs = list(range(len(ds)))
        if args.random:
            random.seed(args.seed)
            random.shuffle(idxs)
        idxs = idxs[: args.num]
        refs_col = ds["captions"]
        paths_col = ds["image_path"]
        return [(paths_col[i], refs_col[i] or None) for i in idxs]

    if args.dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in exts)
        files = files[: args.num] if args.num else files
        return [(str(p), None) for p in files]

    return [(p, None) for p in args.images]


# --------------------------------------------------------------------------- #
# Salida HTML
# --------------------------------------------------------------------------- #
def img_data_uri(img: Image.Image, max_side=384) -> str:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def write_html(out_path, cards, compare, has_refs):
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>VizWiz Captions — demo</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}",
        "h1{font-size:20px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}",
        ".card{background:#1a1d24;border:1px solid #2a2f3a;border-radius:12px;overflow:hidden}",
        ".card img{width:100%;display:block;background:#000}",
        ".body{padding:12px 14px;font-size:14px;line-height:1.45}",
        ".lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8b93a7;margin-top:8px}",
        ".ft{color:#7ee787;font-weight:600}.base{color:#a5b3cc}.ref{color:#c9b6ff}",
        ".fn{font-size:11px;color:#6b7280;margin-top:6px;word-break:break-all}",
        "</style>",
        "<h1>VizWiz-Captions — demo de inferencia (BLIP-base + LoRA)</h1>",
        "<div class='grid'>",
    ]
    for c in cards:
        parts.append("<div class='card'>")
        parts.append(f"<img src='{c['uri']}' alt=''>")
        parts.append("<div class='body'>")
        if compare:
            parts.append("<div class='lbl'>fine-tuned</div>")
            parts.append(f"<div class='ft'>{html.escape(c['ft'])}</div>")
            parts.append("<div class='lbl'>base (sin FT)</div>")
            parts.append(f"<div class='base'>{html.escape(c['base'])}</div>")
        else:
            parts.append(f"<div class='ft'>{html.escape(c['ft'])}</div>")
        if has_refs and c.get("refs"):
            parts.append("<div class='lbl'>referencias humanas</div>")
            for r in c["refs"][:3]:
                parts.append(f"<div class='ref'>• {html.escape(r)}</div>")
        parts.append(f"<div class='fn'>{html.escape(c['fn'])}</div>")
        parts.append("</div></div>")
    parts.append("</div>")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="*", help="rutas de imágenes a captionar")
    p.add_argument("--split", choices=["train", "val", "test"], help="muestrear de un split")
    p.add_argument("--dir", help="captionar todas las imágenes de una carpeta")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--random", action="store_true", help="muestra aleatoria del split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--base-only", action="store_true", help="sin adaptador LoRA")
    p.add_argument("--compare", action="store_true", help="comparar base vs fine-tuned")
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--html", help="ruta de salida de la galería HTML")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    items = gather_images(args)
    if not items:
        p.error("no hay imágenes: pasa rutas, --dir o --split")
    print(f"{len(items)} imágenes | dispositivo: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, has_adapter = load_model(args.model, args.adapter, args.base_only, device)
    compare = args.compare and has_adapter

    paths = [it[0] for it in items]
    refs = [it[1] for it in items]
    imgs = [Image.open(pp).convert("RGB") for pp in paths]

    ft_caps, base_caps = [], []
    bs = args.batch_size
    for s in range(0, len(imgs), bs):
        batch = imgs[s:s + bs]
        ft_caps += caption(model, processor, batch, device, args.num_beams, args.max_new_tokens)
        if compare:
            # desactiva el adaptador para obtener la predicción del modelo base
            with model.disable_adapter():
                base_caps += caption(model, processor, batch, device,
                                     args.num_beams, args.max_new_tokens)

    # salida por consola
    print()
    for i, pp in enumerate(paths):
        name = Path(pp).name
        print(f"[{i+1}] {name}")
        print(f"    caption : {ft_caps[i]}")
        if compare:
            print(f"    base    : {base_caps[i]}")
        if refs[i]:
            print(f"    ref     : {refs[i][0]}")

    # galería HTML
    if args.html:
        cards = []
        for i, pp in enumerate(paths):
            cards.append({
                "uri": img_data_uri(imgs[i]),
                "ft": ft_caps[i],
                "base": base_caps[i] if compare else "",
                "refs": refs[i],
                "fn": Path(pp).name,
            })
        write_html(args.html, cards, compare, has_refs=any(refs))
        print(f"\ngalería HTML escrita en: {args.html}")


if __name__ == "__main__":
    main()

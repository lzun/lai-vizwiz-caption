"""
Comparación de OCR sobre imágenes de VizWiz con texto: EasyOCR (tradicional, GPU) vs
Landing AI ADE (de paga). VizWiz-Captions no trae ground-truth de OCR, así que la
comparación es cualitativa: se vuelca una galería HTML lado a lado con
[imagen · EasyOCR · Landing AI · captions humanas de referencia].

Solo muestrea imágenes con `text_detected = True`.

Clave de Landing AI (VISION_AGENT_API_KEY): se lee de, en este orden,
  1) la variable de entorno,
  2) ~/wizviz/.env,
  3) .env en la raíz del repo (gitignored).
Sin clave, corre solo EasyOCR (usa --no-landingai para forzarlo).

Ejemplos:
    python ocr_compare.py --num 30 --html ocr_compare.html
    python ocr_compare.py --num 8 --no-landingai --html ocr_easyocr.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions  # noqa: E402


# --------------------------------------------------------------------------- #
# Clave de API
# --------------------------------------------------------------------------- #
def resolve_api_key() -> str | None:
    if os.environ.get("VISION_AGENT_API_KEY"):
        return os.environ["VISION_AGENT_API_KEY"]
    for envfile in [Path.home() / "wizviz" / ".env",
                    Path(__file__).resolve().parent / ".env"]:
        if envfile.exists():
            for line in envfile.read_text().splitlines():
                line = line.strip()
                if line.startswith("VISION_AGENT_API_KEY") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# --------------------------------------------------------------------------- #
# Motores OCR
# --------------------------------------------------------------------------- #
def build_easyocr():
    import easyocr
    import torch
    gpu = torch.cuda.is_available()
    print(f"EasyOCR: inicializando (gpu={gpu})...")
    return easyocr.Reader(["en"], gpu=gpu)


def run_easyocr(reader, path: str) -> str:
    # detail=0 -> solo strings. paragraph=True agrupa en líneas legibles.
    # Algunas fotos degradadas rompen el resize interno de EasyOCR (recorte vacío):
    # se captura para que una imagen mala no tumbe toda la corrida.
    for kw in ({"detail": 0, "paragraph": True}, {"detail": 0}):
        try:
            lines = reader.readtext(path, **kw)
            return "\n".join(l.strip() for l in lines if l and l.strip())
        except Exception:
            continue
    return "[EasyOCR: no se pudo procesar la imagen]"


def build_landingai(api_key):
    from landingai_ade import LandingAIADE
    return LandingAIADE(apikey=api_key)


def run_landingai(client, path: str) -> str:
    resp = client.v2.parse(document=Path(path))
    return (resp.markdown or "").strip()


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def data_uri(path, max_side=420):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def write_html(out, records, with_landing):
    css = """
    body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#0f141b;color:#e8ecf2}
    h1{font-family:Georgia,serif;font-size:22px;margin:0 0 4px}
    .sub{color:#95a1b3;font-size:14px;margin:0 0 24px}
    .card{display:grid;grid-template-columns:300px 1fr;gap:20px;background:#161d27;
      border:1px solid #26303c;border-radius:14px;padding:18px;margin-bottom:18px}
    .card img{width:100%;border-radius:8px;background:#000;display:block}
    .fn{font-family:ui-monospace,monospace;font-size:12px;color:#6b7788;margin-top:8px;word-break:break-all}
    .cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
    .box{background:#0f141b;border:1px solid #26303c;border-radius:10px;padding:12px 14px}
    .box h3{margin:0 0 8px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-weight:600}
    .easy h3{color:#4fb0a8}.land h3{color:#e8a34a}.ref h3{color:#a78bfa}
    .txt{font-size:13.5px;line-height:1.5;white-space:pre-wrap;max-height:260px;overflow:auto;
      font-family:ui-monospace,monospace;color:#cdd6e3}
    .ref .txt{font-family:inherit}
    .empty{color:#6b7788;font-style:italic}
    @media(max-width:680px){.card{grid-template-columns:1fr}}
    """
    parts = [f"<!doctype html><meta charset='utf-8'><title>OCR: EasyOCR vs Landing AI</title><style>{css}</style>",
             "<h1>Comparación de OCR — EasyOCR vs Landing&nbsp;AI</h1>",
             f"<p class='sub'>{len(records)} imágenes de VizWiz val con texto detectado · "
             f"{'con' if with_landing else 'sin'} Landing AI</p>"]
    for r in records:
        parts.append("<div class='card'>")
        parts.append(f"<div><img src='{r['uri']}' alt=''><div class='fn'>{html.escape(r['file'])}</div></div>")
        parts.append("<div class='cols'>")
        # EasyOCR
        et = html.escape(r["easyocr"]) if r["easyocr"] else "<span class='empty'>(sin texto)</span>"
        parts.append(f"<div class='box easy'><h3>EasyOCR</h3><div class='txt'>{et}</div></div>")
        # Landing AI
        if with_landing:
            lt = html.escape(r["landingai"]) if r.get("landingai") else "<span class='empty'>(sin texto / n/d)</span>"
            parts.append(f"<div class='box land'><h3>Landing AI (ADE)</h3><div class='txt'>{lt}</div></div>")
        # refs humanas
        refs = "\n".join("• " + rr for rr in (r.get("refs") or []))
        refs = html.escape(refs) if refs else "<span class='empty'>—</span>"
        parts.append(f"<div class='box ref'><h3>Captions humanas</h3><div class='txt'>{refs}</div></div>")
        parts.append("</div></div>")
    Path(out).write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--html", default="ocr_compare.html")
    p.add_argument("--out-json", default="ocr_compare.json")
    p.add_argument("--no-landingai", action="store_true")
    args = p.parse_args()

    root = str(Path(__file__).resolve().parent)
    ds = load_wizviz_captions(root=root, splits=(args.split,),
                              format="grouped", with_images=False)[args.split]
    # solo imágenes con texto detectado
    idxs = [i for i in range(len(ds)) if ds["text_detected"][i]]
    random.seed(args.seed)
    random.shuffle(idxs)
    idxs = idxs[: args.num]
    paths = [ds["image_path"][i] for i in idxs]
    files = [ds["file_name"][i] for i in idxs]
    refs = [ds["captions"][i] for i in idxs]
    print(f"{len(idxs)} imágenes con texto de {args.split}")

    # Landing AI
    api_key = None if args.no_landingai else resolve_api_key()
    with_landing = api_key is not None
    if not args.no_landingai and not with_landing:
        print("[aviso] VISION_AGENT_API_KEY no encontrada -> corro solo EasyOCR.")
        print("        Ponla en ~/wizviz/.env como: VISION_AGENT_API_KEY=tu_clave")

    reader = build_easyocr()
    client = build_landingai(api_key) if with_landing else None

    records = []
    for k, (path, file, ref) in enumerate(zip(paths, files, refs)):
        t0 = time.time()
        easy = run_easyocr(reader, path)
        land = None
        if with_landing:
            try:
                land = run_landingai(client, path)
            except Exception as e:
                land = f"[error Landing AI: {e}]"
        records.append({"file": file, "uri": data_uri(path),
                        "easyocr": easy, "landingai": land, "refs": ref})
        print(f"[{k+1}/{len(paths)}] {file}  ({time.time()-t0:.1f}s)")
        print(f"    easyocr: {easy[:80]!r}")
        if with_landing:
            print(f"    landing: {(land or '')[:80]!r}")

    write_html(args.html, records, with_landing)
    # json sin las imágenes (para reuso)
    slim = [{k: v for k, v in r.items() if k != "uri"} for r in records]
    Path(args.out_json).write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ngalería: {args.html}\njson: {args.out_json}")


if __name__ == "__main__":
    main()

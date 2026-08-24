"""
Precomputa el texto OCR (EasyOCR) de las imágenes necesarias para el experimento de
captioning con OCR (Fase 2) y lo cachea en un JSON {file_name: ocr_text}.

Cubre las imágenes únicas del subset de entrenamiento (primeros --train-pairs pares del
formato exploded) y las --val-images de evaluación. Es reanudable: si el JSON ya existe,
solo procesa las que falten.

Uso:
    python ocr_precompute.py --train-pairs 20000 --val-images 2000 --out ocr_cache_easyocr.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions  # noqa: E402


def run_easyocr(reader, path: str) -> str:
    for kw in ({"detail": 0, "paragraph": True}, {"detail": 0}):
        try:
            lines = reader.readtext(path, **kw)
            return "\n".join(l.strip() for l in lines if l and l.strip())
        except Exception:
            continue
    return ""  # imagen no procesable -> sin OCR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pairs", type=int, default=20000)
    p.add_argument("--val-images", type=int, default=2000)
    p.add_argument("--out", default="ocr_cache_easyocr.json")
    args = p.parse_args()
    root = str(Path(__file__).resolve().parent)

    # imágenes objetivo: {file_name: image_path}
    targets: dict[str, str] = {}
    tr = load_wizviz_captions(root=root, splits=("train",), format="exploded", with_images=False)["train"]
    sub = tr.select(range(min(args.train_pairs, len(tr))))
    for fn, pth in zip(sub["file_name"], sub["image_path"]):
        targets[fn] = pth
    val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
    for fn, pth in zip(val["file_name"][: args.val_images], val["image_path"][: args.val_images]):
        targets[fn] = pth
    print(f"imágenes únicas objetivo: {len(targets)}")

    cache = {}
    out = Path(args.out)
    if out.exists():
        cache = json.loads(out.read_text(encoding="utf-8"))
        print(f"cache existente: {len(cache)} entradas")
    todo = [(fn, pth) for fn, pth in targets.items() if fn not in cache]
    print(f"por procesar: {len(todo)}")
    if not todo:
        print("nada que hacer.")
        return

    import easyocr
    import torch
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    t0 = time.time()
    for i, (fn, pth) in enumerate(todo):
        cache[fn] = run_easyocr(reader, pth)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)} | {el:.0f}s | {(i+1)/el:.1f} img/s", flush=True)
    out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    with_txt = sum(1 for v in cache.values() if v.strip())
    print(f"listo. {len(cache)} imágenes, {with_txt} con texto OCR. -> {out}")


if __name__ == "__main__":
    main()

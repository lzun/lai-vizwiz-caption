"""
Precómputo concurrente (async) del OCR de Landing AI ADE para la Fase 3.

Solo procesa imágenes con `text_detected = True` del subset de entrenamiento
(primeros --train-pairs pares exploded) y de las --val-images de evaluación, para
ahorrar créditos (a las imágenes sin texto no se les llama a la API).

Cachea la salida cruda de ADE (markdown) en un JSON {file_name: markdown}. Reanudable.

Requiere VISION_AGENT_API_KEY (env o ~/wizviz/.env). Cada imagen ~1 crédito.

Uso:
    python ocr_landingai_precompute.py --train-pairs 20000 --val-images 2000 \
        --out ocr_cache_landingai.json --concurrency 8
    python ocr_landingai_precompute.py --limit 3   # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions  # noqa: E402


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


def collect_targets(root, train_pairs, val_images):
    """Imágenes con texto detectado: {file_name: image_path}."""
    targets = {}
    tr = load_wizviz_captions(root=root, splits=("train",), format="exploded",
                              with_images=False)["train"].select(range(train_pairs))
    for fn, pth, td in zip(tr["file_name"], tr["image_path"], tr["text_detected"]):
        if td:
            targets[fn] = pth
    val = load_wizviz_captions(root=root, splits=("val",), format="grouped",
                               with_images=False)["val"]
    for fn, pth, td in zip(val["file_name"][:val_images], val["image_path"][:val_images],
                           val["text_detected"][:val_images]):
        if td:
            targets[fn] = pth
    return targets


async def main_async(args):
    root = str(Path(__file__).resolve().parent)
    api_key = resolve_api_key()
    if not api_key:
        print("ERROR: falta VISION_AGENT_API_KEY (env o ~/wizviz/.env)")
        sys.exit(1)

    targets = collect_targets(root, args.train_pairs, args.val_images)
    print(f"imágenes con texto objetivo: {len(targets)}")

    out = Path(args.out)
    cache = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    print(f"cache existente: {len(cache)}")
    todo = [(fn, p) for fn, p in targets.items() if fn not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"por procesar: {len(todo)} (concurrencia {args.concurrency})")
    if not todo:
        print("nada que hacer."); return

    from landingai_ade import AsyncLandingAIADE
    client = AsyncLandingAIADE(apikey=api_key)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = {"n": 0, "fail": 0}
    t0 = time.time()

    async def worker(fn, path):
        async with sem:
            md = None
            for attempt in range(3):
                try:
                    resp = await client.v2.parse(document=Path(path))
                    md = resp.markdown or ""
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  [fallo] {fn}: {str(e)[:70]}")
                        md = ""
                        done["fail"] += 1
                    else:
                        await asyncio.sleep(2 ** attempt)
            async with lock:
                cache[fn] = md
                done["n"] += 1
                if done["n"] % 50 == 0 or done["n"] == len(todo):
                    out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    el = time.time() - t0
                    print(f"  {done['n']}/{len(todo)} | {el:.0f}s | "
                          f"{done['n']/el:.2f} img/s | fallos {done['fail']}", flush=True)

    await asyncio.gather(*(worker(fn, p) for fn, p in todo))
    out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    with_txt = sum(1 for v in cache.values() if v.strip())
    print(f"listo. {len(cache)} imágenes cacheadas, {with_txt} con contenido. -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pairs", type=int, default=20000)
    p.add_argument("--val-images", type=int, default=2000)
    p.add_argument("--out", default="ocr_cache_landingai.json")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="limita nº de imágenes (smoke)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

"""Evaluación de un modelo BLIP sobre el conjunto de validación COMPLETO (7542 imágenes).
Reusa la generación CORREGIDA con prefijo (fix input_ids[:, :-1] de BLIP.generate) copiada
verbatim de baseline_blip_ocr.evaluate. Guarda las captions (para métricas complementarias)
y reporta CIDEr/BLEU/ROUGE por subconjunto (all/text/notext).

Modos:
  --kind auto     : sin prefijo (modelos autónomos), generación BATCHEADA (rápida).
  --kind prefix   : con prefijo OCR/ADE, generación POR IMAGEN (segura, idéntica al fix).

Ejemplos:
  python reeval_full.py --name blip_ade_vision_full --adapter blip_ade_vision_full \
      --kind prefix --ocr-cache ocr_cache_landingai.json --ocr-clean ade_full
  python reeval_full.py --name blip_scaled4 --adapter blip_scaled4 --kind auto
  python reeval_full.py --name base --adapter "" --kind auto
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics
from baseline_blip_ocr import clean_ocr, MAX_OCR

BLIP = "Salesforce/blip-image-captioning-base"
CAPS_DIR = Path(__file__).resolve().parent / "fullval_caps"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="etiqueta para el archivo de captions")
    p.add_argument("--adapter", default="", help="nombre del dir en ~/wizviz/runs, o vacío para base")
    p.add_argument("--kind", choices=["auto", "prefix"], required=True)
    p.add_argument("--ocr-cache", default="")
    p.add_argument("--ocr-clean", default="ade_full", choices=["none", "ade_full", "ade_textonly"])
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--min-new-tokens", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16, help="solo para --kind auto")
    p.add_argument("--max-eval-samples", type=int, default=0, help="0 = todo val (7542)")
    a = p.parse_args()

    root = str(Path(__file__).resolve().parent)
    device = "cuda"
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(BLIP, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(BLIP, dtype=torch.float16).to(device)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path.home() / "wizviz" / "runs" / a.adapter))
    model.eval()
    tok = proc.tokenizer
    cls = tok.cls_token_id

    val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
    n = len(val["image_id"]) if not a.max_eval_samples else min(a.max_eval_samples, len(val["image_id"]))
    cache = json.loads((Path(root) / a.ocr_cache).read_text(encoding="utf-8")) if a.ocr_cache else {}

    gts, res, is_text = {}, {}, {}
    caps_by_id = {}
    t0 = time.time()

    if a.kind == "prefix":
        with torch.no_grad():
            for k in range(n):
                img = Image.open(val["image_path"][k]).convert("RGB")
                pix = proc(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
                ocr = clean_ocr(cache.get(val["file_name"][k], ""), a.ocr_clean)
                if ocr.strip():
                    ocr_ids = tok(ocr, add_special_tokens=False, truncation=True, max_length=MAX_OCR).input_ids
                    # +[sep]: BLIP.generate hace input_ids[:, :-1]; el extra se descarta (fix verificado)
                    prompt = torch.tensor([[cls] + ocr_ids + [tok.sep_token_id]], device=device)
                    gen = model.generate(pixel_values=pix, input_ids=prompt, num_beams=a.num_beams,
                                         max_new_tokens=a.max_new_tokens, min_new_tokens=a.min_new_tokens)
                    cap = tok.decode(gen[0][1 + len(ocr_ids):], skip_special_tokens=True).strip()
                else:
                    gen = model.generate(pixel_values=pix, num_beams=a.num_beams,
                                         max_new_tokens=a.max_new_tokens, min_new_tokens=a.min_new_tokens)
                    cap = tok.decode(gen[0], skip_special_tokens=True).strip()
                iid = val["image_id"][k]
                gts[iid] = val["captions"][k]; res[iid] = [cap]; caps_by_id[iid] = cap
                is_text[iid] = bool(val["text_detected"][k])
                if (k + 1) % 500 == 0:
                    print(f"  {k+1}/{n} ({(k+1)/(time.time()-t0):.2f} img/s)", flush=True)
    else:  # auto, batcheado
        bs = a.batch_size
        with torch.no_grad():
            for s in range(0, n, bs):
                e = min(s + bs, n)
                imgs = [Image.open(val["image_path"][k]).convert("RGB") for k in range(s, e)]
                pix = proc(images=imgs, return_tensors="pt").pixel_values.to(device, model.dtype)
                gen = model.generate(pixel_values=pix, num_beams=a.num_beams,
                                     max_new_tokens=a.max_new_tokens, min_new_tokens=a.min_new_tokens)
                decoded = tok.batch_decode(gen, skip_special_tokens=True)
                for j, k in enumerate(range(s, e)):
                    iid = val["image_id"][k]
                    cap = decoded[j].strip()
                    gts[iid] = val["captions"][k]; res[iid] = [cap]; caps_by_id[iid] = cap
                    is_text[iid] = bool(val["text_detected"][k])
                if e % 512 == 0 or e == n:
                    print(f"  {e}/{n} ({e/(time.time()-t0):.2f} img/s)", flush=True)

    def subset(pred):
        g = {i: gts[i] for i in gts if pred(i)}; r = {i: res[i] for i in g}
        return caption_metrics(g, r) if g else {}

    results = {
        "all":    (len(gts), caption_metrics(gts, res)),
        "text":   (sum(is_text.values()), subset(lambda i: is_text[i])),
        "notext": (sum(1 - v for v in is_text.values()), subset(lambda i: not is_text[i])),
    }
    CAPS_DIR.mkdir(exist_ok=True)
    (CAPS_DIR / f"{a.name}.json").write_text(
        json.dumps({"caps": caps_by_id, "is_text": {str(k): v for k, v in is_text.items()}},
                   ensure_ascii=False), encoding="utf-8")
    print(f"\n=== {a.name} (val completo, n={len(gts)}) ===")
    for nm, (cnt, m) in results.items():
        print(f"  [{nm:6s} n={cnt:5d}] " + " ".join(f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()

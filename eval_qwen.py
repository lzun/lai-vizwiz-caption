"""
Re-evalúa el adaptador Qwen2.5-VL ya entrenado, forzando salida en INGLÉS (el modelo
bilingüe hacía code-switching a chino, hundiendo el CIDEr). Carga base 4-bit + LoRA,
usa un prompt que exige inglés y bloquea caracteres CJK en la generación. Reporta
métricas por subconjunto y la fracción de captions con caracteres no-ASCII.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from baseline_blip_lora import caption_metrics
import baseline_qwen as bq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default=str(Path.home() / "wizviz" / "runs" / "qwen25vl_20k"))
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--max-eval-samples", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--block-cjk", action="store_true",
                   help="suprimir tokens CJK en la generación (fuerza salida no-china)")
    args = p.parse_args()
    device = "cuda"

    # prompt que fuerza inglés
    bq.PROMPT = ("Provide one short caption describing this image, in the style of a brief "
                 "crowdsourced description. Write in English only.")

    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig)
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256 * 28 * 28,
                                              max_pixels=args.max_side ** 2, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    tok = processor.tokenizer

    # lista de ids de tokens que contienen caracteres CJK (para suppress_tokens)
    suppress = None
    if args.block_cjk:
        def is_cjk(ch):
            o = ord(ch)
            return (0x3000 <= o <= 0x303F or 0x3400 <= o <= 0x9FFF or
                    0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)
        vocab = tok.get_vocab()
        suppress = []
        for t, i in vocab.items():
            s = tok.convert_tokens_to_string([t])
            if any(is_cjk(c) for c in s):
                suppress.append(i)
        print(f"tokens CJK suprimidos: {len(suppress)}")

    val = load_wizviz_captions(root=str(Path(__file__).resolve().parent), splits=("val",),
                               format="grouped", with_images=False)["val"]
    n = min(args.max_eval_samples, len(val))
    collate = bq.make_collate(processor, args.max_side, train=False)

    gts, res, is_text = {}, {}, {}
    nonascii = 0
    t0 = time.time()
    for s in range(0, n, args.eval_batch_size):
        idx = list(range(s, min(s + args.eval_batch_size, n)))
        batch = [{"image": val["image_path"][k], "caption": ""} for k in idx]
        enc = collate(batch).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 num_beams=args.num_beams, suppress_tokens=suppress)
        trimmed = [o[len(i):] for i, o in zip(enc["input_ids"], gen)]
        caps = tok.batch_decode(trimmed, skip_special_tokens=True)
        for k, c in zip(idx, caps):
            c = c.strip()
            if any(ord(ch) > 127 for ch in c):
                nonascii += 1
            gts[val["image_id"][k]] = val["captions"][k]
            res[val["image_id"][k]] = [c]
            is_text[val["image_id"][k]] = bool(val["text_detected"][k])
        if (s + args.eval_batch_size) % 200 < args.eval_batch_size:
            print(f"  {min(s+args.eval_batch_size,n)}/{n} ({min(s+args.eval_batch_size,n)/(time.time()-t0):.2f} img/s)", flush=True)

    print(f"\ncaptions con caracteres no-ASCII (fuga de idioma): {nonascii}/{n} ({100*nonascii/n:.1f}%)")
    print("=== ejemplos ===")
    for i in list(res)[:4]:
        print(f"  pred: {res[i][0]!r}\n  ref : {gts[i][0]!r}")

    def sub(pred):
        g = {i: gts[i] for i in gts if pred(i)}
        return len(g), (caption_metrics(g, {i: res[i] for i in g}) if g else {})
    print(f"\n=== MÉTRICAS (Qwen, inglés forzado) ===")
    for name, (cnt, m) in [("all", (len(gts), caption_metrics(gts, res))),
                           ("text", sub(lambda i: is_text[i])),
                           ("notext", sub(lambda i: not is_text[i]))]:
        print(f"  [{name:6s} n={cnt:4d}] " + " ".join(f"{k}={v}" for k, v in m.items()))


if __name__ == "__main__":
    main()

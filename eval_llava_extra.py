"""Genera captions de LLaVA-1.5-7B (adaptador entrenado) sobre las 2000 val y computa
CIDEr + CLIPScore + BERTScore + SBERT, para situarlo junto a los demás modelos.
Reporta la caption COMPLETA y la truncada al PRIMER ENUNCIADO (separa capacidad de formato)."""
from __future__ import annotations
import sys, time, json, re
from pathlib import Path
import torch
from PIL import Image


def first_sentence(text: str) -> str:
    """Primer enunciado: hasta el primer '.', '!' o '?' seguido de espacio/fin."""
    t = text.strip()
    m = re.search(r"[.!?](\s|$)", t)
    return t[: m.end()].strip() if m else t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics
from eval_metrics_extra import clip_score, bert_score_f1, sbert_sim
import baseline_llava as bl

RUNS = Path.home() / "wizviz" / "runs"


def main():
    root = str(Path(__file__).resolve().parent)
    device = "cuda"
    val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
    n = min(2000, len(val))
    paths = [val["image_path"][i] for i in range(n)]
    refs = [val["captions"][i] for i in range(n)]
    ids = [val["image_id"][i] for i in range(n)]

    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(bl.MODEL, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        bl.MODEL, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(RUNS / "llava15_7b_20k")).eval()
    tok = proc.tokenizer

    caps = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(n):
            img = Image.open(paths[k]).convert("RGB")
            text = bl.build_text(proc, None)
            enc = proc(images=img, text=text, return_tensors="pt").to(device)
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, num_beams=1)
            caps.append(tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{n} ({(k+1)/(time.time()-t0):.2f} img/s)", flush=True)
    del model; import gc; gc.collect(); torch.cuda.empty_cache()

    # Guardar captions crudas (para no regenerar en el futuro)
    out = Path(__file__).resolve().parent / "llava_caps.json"
    out.write_text(json.dumps({"ids": ids, "caps": caps}, ensure_ascii=False), encoding="utf-8")
    print(f"captions guardadas en {out}")

    import statistics
    gts = {ids[i]: refs[i] for i in range(n)}

    def evaluate(variant_caps, label):
        wlen = statistics.mean(len(c.split()) for c in variant_caps)
        res = {ids[i]: [variant_caps[i]] for i in range(n)}
        cider = caption_metrics(gts, res).get("CIDEr")
        cs = clip_score(paths, variant_caps, device)
        bs = bert_score_f1(variant_caps, refs)
        ss = sbert_sim(variant_caps, refs, device)
        print(f"\n=== LLaVA-1.5-7B · {label} (2000 val) ===")
        print(f"palabras/caption media: {wlen:.1f}")
        print(f"CIDEr={cider}  CLIPScore={cs:.2f}  BERTScore={100*bs:.2f}  SBERT={100*ss:.2f}")

    evaluate(caps, "completa")
    caps_trunc = [first_sentence(c) for c in caps]
    evaluate(caps_trunc, "primer enunciado")
    # 3 ejemplos completa -> truncada
    print("\nEjemplos (completa -> truncada):")
    for i in range(3):
        print(f"  full : {caps[i]}")
        print(f"  trunc: {caps_trunc[i]}")


if __name__ == "__main__":
    main()

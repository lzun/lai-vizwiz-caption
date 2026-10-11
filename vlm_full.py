"""Evaluación de los VLM alternativos (GIT, Qwen2.5-VL, LLaVA-1.5-7B) sobre val COMPLETO.
Resumible e incremental: guarda fullval_caps/<model>.json (id->cap) cada 200 imágenes y
al reanudar salta los ids ya hechos (necesario porque LLaVA ~4h excede el tope de 2h).
Al completar las 7542 imágenes imprime CIDEr/BLEU/ROUGE por subconjunto.

Uso: python vlm_full.py --model {git|qwen|llava}
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_lora import caption_metrics

RUNS = Path.home() / "wizviz" / "runs"
CAPS_DIR = Path(__file__).resolve().parent / "fullval_caps"
CAPS_DIR.mkdir(exist_ok=True)


def load_git(device):
    from transformers import AutoProcessor, GitForCausalLM
    from peft import PeftModel
    name = "microsoft/git-large-coco"
    proc = AutoProcessor.from_pretrained(name)
    model = GitForCausalLM.from_pretrained(name, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(model, str(RUNS / "git_coco_full_r32")).eval()

    def gen(path):
        img = Image.open(path).convert("RGB")
        pix = proc(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
        with torch.no_grad():
            g = model.generate(pixel_values=pix, num_beams=3, max_length=40)
        return proc.batch_decode(g, skip_special_tokens=True)[0].strip()
    return gen


def load_qwen(device):
    import baseline_qwen as bq
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    name = "Qwen/Qwen2.5-VL-3B-Instruct"
    bq.PROMPT = ("Describe this image in one short English caption, in the style of a brief "
                 "crowdsourced description. Always answer in English only.")
    proc = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=448 ** 2, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(RUNS / "qwen25vl_20k")).eval()
    tok = proc.tokenizer

    def is_cjk(ch):
        o = ord(ch)
        return (0x3000 <= o <= 0x303F or 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)
    suppress = [i for t, i in tok.get_vocab().items() if any(is_cjk(c) for c in tok.convert_tokens_to_string([t]))]
    collate = bq.make_collate(proc, 448, train=False)

    def gen(path):
        enc = collate([{"image": path, "caption": ""}]).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, num_beams=1, suppress_tokens=suppress)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen


def load_llava(device):
    import baseline_llava as bl
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(bl.MODEL, use_fast=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        bl.MODEL, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(RUNS / "llava15_7b_20k")).eval()
    tok = proc.tokenizer

    def gen(path):
        img = Image.open(path).convert("RGB")
        text = bl.build_text(proc, None)
        enc = proc(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, num_beams=1)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["git", "qwen", "llava"], required=True)
    a = ap.parse_args()
    root = str(Path(__file__).resolve().parent)
    device = "cuda"
    val = load_wizviz_captions(root=root, splits=("val",), format="grouped", with_images=False)["val"]
    ids = [val["image_id"][i] for i in range(len(val["image_id"]))]
    paths = {val["image_id"][i]: val["image_path"][i] for i in range(len(ids))}
    refs = {val["image_id"][i]: val["captions"][i] for i in range(len(ids))}
    istext = {val["image_id"][i]: bool(val["text_detected"][i]) for i in range(len(ids))}

    out = CAPS_DIR / f"{a.model}.json"
    done = {}
    if out.exists():
        done = json.loads(out.read_text(encoding="utf-8")).get("caps", {})
    todo = [i for i in ids if str(i) not in done]
    print(f"{a.model}: {len(done)} hechas, {len(todo)} por hacer", flush=True)

    if todo:
        gen = {"git": load_git, "qwen": load_qwen, "llava": load_llava}[a.model](device)
        t0 = time.time()
        for n, iid in enumerate(todo):
            try:
                done[str(iid)] = gen(paths[iid])
            except Exception as e:
                done[str(iid)] = ""
                print(f"  [err {iid}]: {str(e)[:80]}", flush=True)
            if (n + 1) % 200 == 0 or n + 1 == len(todo):
                out.write_text(json.dumps({"caps": done,
                    "is_text": {str(k): v for k, v in istext.items()}}, ensure_ascii=False), encoding="utf-8")
                print(f"  {n+1}/{len(todo)} ({(n+1)/(time.time()-t0):.2f} img/s) [guardado]", flush=True)

    if len(done) < len(ids):
        print(f"incompleto: {len(done)}/{len(ids)} — reencolar para continuar", flush=True)
        return

    # métricas por subconjunto (todas las imágenes hechas)
    def sub(keep):
        g = {i: refs[i] for i in ids if keep(i)}
        r = {i: [done[str(i)]] for i in g}
        return len(g), (caption_metrics(g, r) if g else {})
    for nm, keep in [("all", lambda i: True), ("text", lambda i: istext[i]), ("notext", lambda i: not istext[i])]:
        c, m = sub(keep)
        print(f"  [{a.model} {nm:6s} n={c:5d}] " + " ".join(f"{k}={v}" for k, v in m.items()), flush=True)


if __name__ == "__main__":
    main()

"""Verifica el bug de BLIP.generate (descarta input_ids[:, :-1]) en la ruta con prefijo.
Compara el recorte BUGGY vs el FIX en un ejemplo real con ADE, e imprime la estructura de tokens."""
from __future__ import annotations
import sys, json
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_ocr import clean_ocr, MAX_OCR

root = Path(__file__).resolve().parent
device = "cuda"
from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import PeftModel
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", dtype=torch.float16).to(device)
model = PeftModel.from_pretrained(model, str(Path.home() / "wizviz/runs/blip_ade_vision_full")).eval()
tok = proc.tokenizer
print(f"cls={tok.cls_token_id} bos={model.config.text_config.bos_token_id} sep={tok.sep_token_id} pad={tok.pad_token_id}")

val = load_wizviz_captions(root=str(root), splits=("val",), format="grouped", with_images=False)["val"]
ade = json.loads((root / "ocr_cache_landingai.json").read_text(encoding="utf-8"))

# toma las primeras imágenes con ADE que tengan texto
shown = 0
for k in range(len(val["file_name"])):
    fn = val["file_name"][k]
    ocr = clean_ocr(ade.get(fn, ""), "ade_full")
    if not ocr.strip():
        continue
    img = Image.open(val["image_path"][k]).convert("RGB")
    pix = proc(images=img, return_tensors="pt").pixel_values.to(device, model.dtype)
    ocr_ids = tok(ocr, add_special_tokens=False, truncation=True, max_length=MAX_OCR).input_ids

    # BUGGY: prompt=[cls]+ocr_ids, recorte [1+len:]
    prompt_b = torch.tensor([[tok.cls_token_id] + ocr_ids], device=device)
    g_b = model.generate(pixel_values=pix, input_ids=prompt_b, num_beams=3, max_new_tokens=30, min_new_tokens=5)
    cap_b = tok.decode(g_b[0][1 + len(ocr_ids):], skip_special_tokens=True).strip()

    # FIX: prompt=[cls]+ocr_ids+[sep] (el token extra se descarta por generate), recorte [1+len:]
    prompt_f = torch.tensor([[tok.cls_token_id] + ocr_ids + [tok.sep_token_id]], device=device)
    g_f = model.generate(pixel_values=pix, input_ids=prompt_f, num_beams=3, max_new_tokens=30, min_new_tokens=5)
    cap_f = tok.decode(g_f[0][1 + len(ocr_ids):], skip_special_tokens=True).strip()

    print(f"\n[{fn}] len(ocr_ids)={len(ocr_ids)}")
    print(f"  prefijo en salida BUGGY: {g_b[0][:1+len(ocr_ids)+1].tolist()[-3:]} ... len_out={g_b.shape[1]}")
    print(f"  BUGGY: {cap_b!r}")
    print(f"  FIX  : {cap_f!r}")
    shown += 1
    if shown >= 5:
        break

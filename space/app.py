"""
WizViz Caption — servicio web de descripción de imágenes (prototipo).

Imagen de entrada -> descripción en inglés, con el mejor modelo del proyecto:
BLIP-base + LoRA texto+visión (r=32, 4 épocas, train completo de VizWiz-Captions),
CIDEr 74.6 en val. Es autónomo: corre local, en CPU o GPU, sin servicios de pago.

Modo opcional "ADE" (CIDEr 78.7): si hay VISION_AGENT_API_KEY y el adaptador
`blip_ade_vision_full`, la imagen se manda antes a Landing AI ADE y su descripción
se usa como prefijo del decoder. Cuesta ~1 crédito por imagen y añade latencia.

Pensado para Hugging Face Spaces (SDK Gradio). Gradio expone además una API HTTP
(ver "Use via API" al pie de la página, o el cliente `gradio_client`).

Variables de entorno:
    WIZVIZ_ADAPTER       ruta local o repo del Hub del adaptador autónomo
                         (default: ./adapters/blip_scaled4)
    WIZVIZ_ADE_ADAPTER   ruta local o repo del Hub del adaptador ADE
                         (default: ./adapters/blip_ade_vision_full)
    VISION_AGENT_API_KEY clave de Landing AI (habilita el modo ADE)
    HF_TOKEN             solo si los adaptadores están en un repo privado del Hub

Local (dentro de WSL, con el venv ~/wizviz/.venv):
    python space/app.py            # -> http://localhost:7860
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

# ZeroGPU (Hugging Face Spaces): `spaces` debe importarse antes que torch, y exige al
# menos una función con @spaces.GPU. Fuera de HF el paquete no hace falta: no-op.
try:
    import spaces

    gpu = spaces.GPU(duration=30)
except ImportError:
    def gpu(fn):
        return fn

import gradio as gr
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

HERE = Path(__file__).resolve().parent
BASE_MODEL = "Salesforce/blip-image-captioning-base"
ADAPTER = os.environ.get("WIZVIZ_ADAPTER", str(HERE / "adapters" / "blip_scaled4"))
ADE_ADAPTER = os.environ.get("WIZVIZ_ADE_ADAPTER", str(HERE / "adapters" / "blip_ade_vision_full"))
ADE_KEY = os.environ.get("VISION_AGENT_API_KEY", "").strip()

# Mismos hiperparámetros de generación que en la evaluación (baseline_blip_ocr.py)
NUM_BEAMS = 3
MAX_NEW_TOKENS = 30
MIN_NEW_TOKENS = 5
MAX_OCR = 32  # tokens máximos del prefijo ADE (igual que en el entrenamiento)
MAX_SIDE = 1600  # reduce fotos enormes antes de mandarlas a ADE

ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
# En ZeroGPU la GPU solo existe dentro de las funciones @gpu, pero el modelo se mueve a
# "cuda" al arrancar (spaces difiere el traslado hasta la primera llamada).
DEVICE = "cuda" if ZERO_GPU or torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# --------------------------------------------------------------------------- #
# Carga del modelo (una vez, al arrancar)
# --------------------------------------------------------------------------- #
def _adapter_available(ref: str) -> bool:
    """Ruta local existente, o algo con forma de repo del Hub (usuario/nombre)."""
    if Path(ref).exists():
        return True
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", ref))


def load():
    from peft import PeftModel

    processor = BlipProcessor.from_pretrained(BASE_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=DTYPE)
    # torch_device="cpu": PEFT detecta "cuda" (en ZeroGPU la emulación dice que hay GPU) y
    # carga los safetensors directo ahí, lo que falla con "No CUDA GPUs are available".
    # Se carga en CPU y luego .to(DEVICE) mueve todo el modelo.
    model = PeftModel.from_pretrained(model, ADAPTER, adapter_name="auto", torch_device="cpu")
    ade_ready = False
    if ADE_KEY and _adapter_available(ADE_ADAPTER):
        try:
            model.load_adapter(ADE_ADAPTER, adapter_name="ade", torch_device="cpu")
            ade_ready = True
        except Exception as e:  # el modo ADE es opcional: no tumbar la app
            print(f"[aviso] no se pudo cargar el adaptador ADE ({ADE_ADAPTER}): {e}")
    model.set_adapter("auto")
    model.to(DEVICE).eval()
    return processor, model, ade_ready


processor, model, ADE_READY = load()
print(f"modelo listo en {DEVICE} | modo ADE: {'sí' if ADE_READY else 'no'}", flush=True)


# --------------------------------------------------------------------------- #
# ADE (Landing AI) — opcional
# --------------------------------------------------------------------------- #
def clean_ade(text: str) -> str:
    """Igual que clean_ocr(mode='ade_full') del entrenamiento."""
    t = re.sub(r"<!--.*?-->", " ", text or "", flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ade_describe(img: Image.Image) -> str:
    from landingai_ade import LandingAIADE

    im = img.copy()
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "img.jpg"
        im.save(path, format="JPEG", quality=90)
        resp = LandingAIADE(apikey=ADE_KEY).v2.parse(document=path)
    return clean_ade(resp.markdown or "")


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
@gpu
@torch.no_grad()
def generate(img: Image.Image, prefix: str = "", adapter: str = "auto") -> str:
    # El adaptador se elige aquí dentro: en ZeroGPU esta función corre en otro proceso,
    # así que los cambios de estado hechos fuera no le llegan.
    model.set_adapter(adapter)
    tok = processor.tokenizer
    pix = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)
    kw = dict(num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS)
    if prefix:
        ids = tok(prefix, add_special_tokens=False, truncation=True, max_length=MAX_OCR).input_ids
        # BlipForConditionalGeneration.generate descarta el último token de input_ids
        # (espera que termine en [SEP]); sin este [SEP] de relleno se pierde el último
        # token del prefijo y el corte de abajo se come la primera palabra generada.
        prompt = torch.tensor([[tok.cls_token_id] + ids + [tok.sep_token_id]], device=DEVICE)
        out = model.generate(pixel_values=pix, input_ids=prompt, **kw)
        return tok.decode(out[0][1 + len(ids):], skip_special_tokens=True).strip()
    out = model.generate(pixel_values=pix, **kw)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def ade_prefix_used(ade_text: str) -> str:
    """Lo que realmente ve el decoder: el texto de ADE truncado a MAX_OCR tokens."""
    tok = processor.tokenizer
    ids = tok(ade_text, add_special_tokens=False, truncation=True, max_length=MAX_OCR).input_ids
    return tok.decode(ids)


def run(img: Image.Image | None, use_ade: bool, compare: bool = False) -> dict:
    """Núcleo común de la API y la UI. Con compare=True y ADE activo, genera también
    la descripción autónoma (no gasta créditos extra: ADE se llama una sola vez)."""
    if img is None:
        raise gr.Error("Sube una imagen primero.")
    img = img.convert("RGB")
    hw = "ZeroGPU" if ZERO_GPU else DEVICE
    r = {"ade_text": "", "auto": ""}
    t_ade = 0.0
    if use_ade and ADE_READY:
        t0 = time.time()
        try:  # ADE se llama fuera de @gpu para no gastar cuota de GPU esperando la red
            r["ade_text"] = ade_describe(img)
        except Exception as e:
            print(f"[aviso] ADE falló: {e}", flush=True)
            gr.Warning(f"ADE falló ({str(e)[:80]}); uso el modelo autónomo.")
        t_ade = time.time() - t0
    t0 = time.time()
    if r["ade_text"]:
        r["caption"] = _cap(generate(img, prefix=r["ade_text"], adapter="ade"))
        if compare:
            r["auto"] = _cap(generate(img))
        r["info"] = f"BLIP + ADE · {hw} · ADE {t_ade:.1f} s + modelo {time.time() - t0:.1f} s"
    else:
        r["caption"] = _cap(generate(img))
        r["info"] = f"BLIP autónomo · {hw} · {time.time() - t0:.1f} s"
    return r


def describe(img: Image.Image | None, use_ade: bool = False):
    """Endpoint público /describe: (descripción, info)."""
    r = run(img, use_ade)
    return r["caption"], r["info"]


def describe_ui(img: Image.Image | None, use_ade: bool = False):
    """UI: además muestra la comparación con el autónomo y el texto de ADE."""
    r = run(img, use_ade, compare=True)
    used = bool(r["ade_text"])
    return (
        r["caption"],
        r["info"],
        gr.Textbox(value=r["auto"], visible=used),
        gr.Accordion(visible=used),
        ade_prefix_used(r["ade_text"]) if used else "",
        r["ade_text"],
    )


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
DESC = (
    "Sube una foto (idealmente tomada por una persona ciega o con baja visión, como en "
    "VizWiz) y el modelo genera una descripción breve en inglés. "
    "Modelo: BLIP-base + LoRA texto+visión fine-tuneado en VizWiz-Captions "
    "(CIDEr 74.6 en validación)."
)

examples_dir = HERE / "examples"
examples = sorted(str(p) for p in examples_dir.glob("*.jpg")) if examples_dir.exists() else []

with gr.Blocks(title="WizViz Caption") as demo:
    gr.Markdown("# WizViz Caption\n" + DESC)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Imagen", sources=["upload", "webcam", "clipboard"])
            ade_box = gr.Checkbox(
                label="Usar Landing AI ADE (más preciso con texto; ~1 crédito por imagen)",
                value=False, visible=ADE_READY,
            )
            btn = gr.Button("Describir", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Descripción", lines=3, buttons=["copy"])
            info = gr.Markdown()
            auto_out = gr.Textbox(label="Sin ADE (modelo autónomo), para comparar",
                                  lines=2, visible=False)
            with gr.Accordion("Lo que leyó ADE", open=True, visible=False) as ade_acc:
                ade_used = gr.Textbox(
                    label=f"Prefijo que recibe el modelo (primeros {MAX_OCR} tokens)", lines=3)
                ade_full = gr.Textbox(label="Salida completa de ADE (limpia)", lines=6,
                                      buttons=["copy"])
    ui_outputs = [out, info, auto_out, ade_acc, ade_used, ade_full]

    # API pública estable: (descripción, info). Va en un botón oculto para que el
    # endpoint conserve el esquema de los componentes; se registra antes que todo lo
    # demás para quedarse con el nombre /describe.
    gr.Button(visible=False).click(describe, inputs=[inp, ade_box], outputs=[out, info],
                                   api_name="describe")
    btn.click(describe_ui, inputs=[inp, ade_box], outputs=ui_outputs,
              api_name="describe_ui", api_visibility="private")
    inp.upload(describe_ui, inputs=[inp, ade_box], outputs=ui_outputs,
               api_name="describe_on_upload", api_visibility="private")
    if examples:
        ex = gr.Examples(examples=examples, inputs=inp, cache_examples=False,
                         api_name="example", api_visibility="private")
        # Tras cargar el ejemplo, se describe respetando la casilla de ADE.
        ex.load_input_event.then(describe_ui, inputs=[inp, ade_box], outputs=ui_outputs,
                                 api_visibility="private")

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

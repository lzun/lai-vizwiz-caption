---
title: WizViz Caption
emoji: 👁️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
python_version: "3.12"
pinned: false
license: cc-by-4.0
short_description: Descripción de imágenes estilo VizWiz (BLIP + LoRA)
---

# WizViz Caption — servicio web

Imagen → descripción breve en inglés, con el mejor modelo autónomo del proyecto:
**BLIP-base + LoRA texto+visión** (r=32, alpha=64, train completo, 4 épocas) fine-tuneado
en VizWiz-Captions. **CIDEr 74.6** en validación. Corre en CPU o GPU, sin servicios de pago.

Modo opcional **BLIP + Landing AI ADE** (CIDEr 78.7 en val): solo se activa si el
servidor tiene `VISION_AGENT_API_KEY` y el adaptador `blip_ade_vision_full`. Cuesta
~1 crédito de Landing AI por imagen y tarda ~7–10 s. Con ADE activo, la interfaz muestra
además la descripción **sin ADE** (para comparar; no gasta créditos extra) y un panel
**"Lo que leyó ADE"** con el prefijo que recibe el modelo (primeros 32 tokens) y la salida
completa de ADE. La API `/describe` no cambia: sigue devolviendo `(descripción, info)`.

## Estructura

```
space/
├── app.py               # servicio Gradio (UI + API HTTP)
├── requirements.txt
├── run_local.sh         # arranque local en WSL
├── client_example.py    # ejemplo de llamada a la API
├── examples/            # 6 imágenes de val de VizWiz (CC BY 4.0)
└── adapters/            # NO versionado en git (pesos, ver abajo)
    ├── blip_scaled4/            # autónomo (CIDEr 74.6) — obligatorio
    └── blip_ade_vision_full/    # ADE (CIDEr 78.7) — opcional
```

Los adaptadores se copian desde WSL (`~/wizviz/runs/<nombre>/adapter_config.json` y
`adapter_model.safetensors`, ~14 MB cada uno).

## Ejecutar en local (WSL)

```bash
wsl -e bash /mnt/c/Users/green/Documents/GitHub/wizviz-caption/space/run_local.sh
```

Abre <http://localhost:7860>. Si `~/wizviz/.env` tiene `VISION_AGENT_API_KEY`, aparece
la casilla del modo ADE.

## API

Gradio expone el endpoint `/describe` (ver "Usar vía API" al pie de la página).

```bash
python client_example.py foto.jpg                                   # local
python client_example.py foto.jpg --url <usuario>/wizviz-caption    # Space
```

## Desplegar en Hugging Face Spaces

Las cuentas gratuitas **no** pueden alojar Spaces Gradio en `cpu-basic` (desde 2026 eso
requiere PRO; `hf repo create` devuelve `402 Payment Required`). La opción gratuita es
**ZeroGPU**: crea el Space desde la web (SDK Gradio, hardware *ZeroGPU*) y sube los archivos:

```bash
hf auth login
hf upload <usuario>/wizviz-caption space/ . --repo-type space --exclude "adapters/blip_ade_vision_full/*"
```

- `app.py` ya es compatible con ZeroGPU: importa `spaces`, mueve el modelo a `cuda` al
  arrancar y la generación va en una función `@spaces.GPU` (sin ella el Space falla con
  *"No @spaces.GPU function detected during startup"*). En local el decorador no hace nada.
- Requisitos de ZeroGPU: cuenta con email verificado y >30 días de antigüedad (máx. 2 Spaces
  gratis); torch 2.8+ (fijado a 2.11.0) y Python 3.12 (`python_version` arriba).
- Cuota diaria **por visitante**: 2 min sin sesión, 5 min con cuenta gratuita. Cada
  descripción usa ~0.5 s de GPU, así que alcanza para cientos de imágenes al día.
- **Modo ADE en un Space público:** si se configura `VISION_AGENT_API_KEY` como *secret*,
  cualquier visitante gasta créditos de la cuenta. Recomendado: dejarlo sin clave (solo
  modo autónomo) o hacer el Space privado. Para habilitarlo, sube también
  `adapters/blip_ade_vision_full/` (quita el `--exclude`).
- Alternativa: subir los adaptadores a un repo de modelo del Hub y apuntar
  `WIZVIZ_ADAPTER` / `WIZVIZ_ADE_ADAPTER` (variables del Space) a `<usuario>/<repo>`.

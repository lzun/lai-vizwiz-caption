#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AVENIDA ADE + VISIÓN — Fase B (train COMPLETO). ⚠️ GASTA CRÉDITOS.
#
# La Fase A confirmó que ADE + visión se SUMAN (Δ +4.5 CIDEr en imágenes con texto).
# Esta fase escala ADE al train completo para ver si el delta se mantiene.
#
# ⚠️ PRESUPUESTO: precomputar ADE para las ~14.700 imágenes-con-texto del train
#    completo cuesta ~11.800 créditos (ya hay ~2.900 cacheadas → faltan ~11.800).
#    NO ejecutar sin ese presupuesto. El precómputo es reanudable (guarda cada 50).
#
# Control para comparar = el modelo solo-visión de 74.60 (~/wizviz/runs/blip_scaled4).
# Para su desglose por texto/sin-texto, re-evaluar ese adaptador:
#    python baseline_blip_ocr.py --no-train --target-modules query,value,qkv --lora-r 32 \
#      --output-dir ~/wizviz/runs/blip_scaled4 --max-eval-samples 2000
#
# Ejecutar desde Windows:
#   wsl -e bash /mnt/c/Users/green/Documents/GitHub/wizviz-caption/run_ade_vision_phaseB.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
VPY="$HOME/wizviz/.venv/bin/python"
REPO="/mnt/c/Users/green/Documents/GitHub/wizviz-caption"
LOG="$HOME/wizviz/run_ade_vision_B.log"
cd "$REPO"
echo "===== ADE+VISIÓN Fase B start: $(date) =====" > "$LOG"

echo ">>> [1/2] Precómputo ADE del train COMPLETO (reanudable, gasta créditos)" | tee -a "$LOG"
"$VPY" ocr_landingai_precompute.py --train-pairs 100575 --val-images 2000 \
  --concurrency 10 --out ocr_cache_landingai.json >> "$LOG" 2>&1

echo ">>> [2/2] Entrenar full + visión + ADE (r=32, 4 épocas)" | tee -a "$LOG"
"$VPY" baseline_blip_ocr.py --ocr --ocr-cache ocr_cache_landingai.json --ocr-clean ade_full \
  --target-modules "query,value,qkv" --lora-r 32 --lora-alpha 64 \
  --max-train-samples 100575 --max-eval-samples 2000 --epochs 4 \
  --batch-size 12 --grad-accum 1 --save-steps 800 \
  --output-dir "$HOME/wizviz/runs/blip_ade_vision_full" >> "$LOG" 2>&1

echo "===== ADE+VISIÓN Fase B end: $(date) =====" >> "$LOG"
echo "Comparar CIDEr 'con texto' vs el control solo-visión (blip_scaled4, 74.60 global)."

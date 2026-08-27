#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# AVENIDA ADE + VISIÓN — Fase A (sin gastar créditos nuevos)
#
# Pregunta: ¿la descripción de ADE (prefijo) SIGUE aportando cuando el modelo ya
# tiene fine-tuning del encoder de visión, o el vision-scaling ya capturó su valor?
#
# A/B controlado sobre el MISMO subset de 20k pares (para el que YA tenemos ADE
# cacheado en ocr_cache_landingai.json → 0 créditos), ambos brazos con LoRA
# texto+visión (r=32, la config ganadora):
#   * control-visión : LoRA(texto+visión), SIN ADE
#   * ade-visión     : LoRA(texto+visión), CON prefijo ADE
#
# Comparar CIDEr desglosado por texto/sin-texto. Referencia (hilo OCR, TEXTO-ONLY):
#   Fase 3 control 51.68 (texto)  vs  ADE-prefijo 54.84 (texto)  → ADE sumaba +3.2
#   Aquí veremos si ese +3.2 se mantiene, se encoge o desaparece con visión.
#
# Ejecutar desde Windows:
#   wsl -e bash /mnt/c/Users/green/Documents/GitHub/wizviz-caption/run_ade_vision_phaseA.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
VPY="$HOME/wizviz/.venv/bin/python"
REPO="/mnt/c/Users/green/Documents/GitHub/wizviz-caption"
LOG="$HOME/wizviz/run_ade_vision_A.log"
EPOCHS="${EPOCHS:-2}"   # 2 para tantear rápido; export EPOCHS=4 para el número final
cd "$REPO"
echo "===== ADE+VISIÓN Fase A start: $(date) | epochs=$EPOCHS =====" > "$LOG"

echo ">>> [1/2] control-visión (texto+visión, SIN ADE)" | tee -a "$LOG"
"$VPY" baseline_blip_ocr.py \
  --target-modules "query,value,qkv" --lora-r 32 --lora-alpha 64 \
  --max-train-samples 20000 --max-eval-samples 2000 --epochs "$EPOCHS" \
  --batch-size 12 --grad-accum 1 --save-steps 800 \
  --output-dir "$HOME/wizviz/runs/blip_ctrl_vision" >> "$LOG" 2>&1

echo ">>> [2/2] ade-visión (texto+visión, CON prefijo ADE)" | tee -a "$LOG"
"$VPY" baseline_blip_ocr.py --ocr --ocr-cache ocr_cache_landingai.json --ocr-clean ade_full \
  --target-modules "query,value,qkv" --lora-r 32 --lora-alpha 64 \
  --max-train-samples 20000 --max-eval-samples 2000 --epochs "$EPOCHS" \
  --batch-size 12 --grad-accum 1 --save-steps 800 \
  --output-dir "$HOME/wizviz/runs/blip_ade_vision" >> "$LOG" 2>&1

echo "===== ADE+VISIÓN Fase A end: $(date) =====" >> "$LOG"
echo "Resultados en $LOG (busca 'MÉTRICAS')"

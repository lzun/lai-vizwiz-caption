#!/usr/bin/env bash
# Arranca el servicio en local (WSL, venv ~/wizviz/.venv). Abre http://localhost:7860 en Windows.
# Para habilitar el modo ADE, la clave se toma de ~/wizviz/.env si existe.
set -e
cd "$(dirname "$0")"
if [ -f "$HOME/wizviz/.env" ]; then set -a; . "$HOME/wizviz/.env"; set +a; fi
exec "$HOME/wizviz/.venv/bin/python" app.py

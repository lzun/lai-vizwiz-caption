"""
Ejemplo de uso del servicio como API (local o en Hugging Face Spaces).

    pip install gradio_client
    python client_example.py foto.jpg                       # servidor local
    python client_example.py foto.jpg --url usuario/wizviz-caption   # Space del Hub
"""
import argparse

from gradio_client import Client, handle_file

p = argparse.ArgumentParser()
p.add_argument("images", nargs="+")
p.add_argument("--url", default="http://localhost:7860/")
p.add_argument("--ade", action="store_true", help="usar el modo ADE (si el servidor lo tiene)")
args = p.parse_args()

client = Client(args.url, verbose=False)
for path in args.images:
    caption, info = client.predict(handle_file(path), args.ade, api_name="/describe")
    print(f"{path}: {caption}   [{info}]")

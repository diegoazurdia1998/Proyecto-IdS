"""
Lista los modelos disponibles con tu API key de Groq.
Uso: python scripts/list_groq_models.py
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY no está configurada en .env")
    raise SystemExit(1)

print(f"🔑 API key: {api_key[:8]}...{api_key[-4:]}")
print("📋 Consultando modelos disponibles...\n")

client = Groq(api_key=api_key)
try:
    models = client.models.list()
except Exception as e:
    print(f"❌ Error al consultar modelos: {e}")
    raise SystemExit(1)

print("Modelos disponibles con tu API key:\n")
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  ✅ {m.id}")
    if hasattr(m, 'owned_by') and m.owned_by:
        print(f"      owned_by: {m.owned_by}")

print(f"\nTotal: {len(models.data)} modelos")